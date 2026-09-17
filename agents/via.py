import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value
from functools import partial


class VIA(flax.struct.PyTreeNode):
    """Unified variant: a single time-conditioned critic V(s, x, t).

Losses:
  1) Environment Bellman on the t=1 slice:
     V(s, a, 1) <- r + gamma^h * mean_z V_bar(s', z, 0)
     (noise marginalization: E_z[V(s', z, 0)] = E_{a'~pi}[Q(s', a')],
      so the target requires no policy rollout)
  2) Flow-time TD on the interior:
     V(s, x_t, t) <- sg V(s, x_t + v*delta, t + delta)
     When a step crosses the boundary, _flow_step pins t_next = 1 and clips x,
     so the same network's t=1 slice is consumed as the target automatically.
  3) Actor: CFM BC + advantage-normalized guidance at a lookahead point drawn
     from the same step law as the critic's TD transitions.

Training is strictly independent of flow_steps (F is inference-only).
"""
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    TIME_FEAT_FREQS = 32   # time features = 2 * 32 = 64 dims

    @classmethod
    def _time_features(cls, t):
        # linear frequencies pi..32pi (exponential 2^k becomes noise-like on t in [0,1] for large k)
        freqs = (jnp.arange(cls.TIME_FEAT_FREQS) + 1.0) * jnp.pi
        ang = t * freqs
        return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)

    @classmethod
    def _with_time(cls, x, t):
        if jnp.ndim(t) == 0:
            t = jnp.full((*x.shape[:-1], 1), t)
        return jnp.concatenate([x, cls._time_features(t)], axis=-1)

    def _v(self, module, obs, x, t, params=None):
        return self.network.select(module)(obs, actions=self._with_time(x, t), params=params)

    def _agg(self, qs):
        if self.config['q_agg'] == 'min':
            return qs.min(axis=0)
        return qs.mean(axis=0)




    def _sample_v0_noise(self, rng, action_shape):
        K = int(self.config['num_v0_samples'])
        assert K % 2 == 0
        m = K // 2
        B, A = action_shape
        g_rng, r_rng = jax.random.split(rng)
        g = jax.random.normal(g_rng, (m, B, A))
        qs = []
        for i in range(m):
            v = g[i]
            for j in range(i - (i % A), i):           
                v = v - jnp.sum(v * qs[j], axis=-1, keepdims=True) * qs[j]
            qs.append(v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-8))
        dirs = jnp.stack(qs, axis=0)                   # (m, B, A)
        radius = jnp.sqrt(float(A))
        half = radius * dirs
        return jnp.concatenate([half, -half], axis=0)  # (K, B, A)

    def _sample_step(self, rng, t):
        """delta ~ Unif(0, 1-t+eps); snap to the boundary (t=1) past the remaining time.
        P(bound | t) = eps / (1-t+eps): ~eps at t=0, -> 1 as t -> 1.
        Conditional on non-bound, delta ~ Unif(0, 1-t)."""
        remain = 1.0 - t
        u = jax.random.uniform(rng, t.shape)
        raw = u * (remain + self.config['margin'])   # 0.1
        bound = raw >= remain
        delta = jnp.where(bound, remain, raw)
        return delta, bound
    def _flow_step(self, x, t, vel, delta, bound):
        """Advance (x, t) by delta. On boundary: pin t=1 and clip x like an executed action."""
        x_next = x + vel * delta
        x_next = jnp.where(bound, jnp.clip(x_next, -1, 1), x_next)
        t_next = jnp.where(bound, 1.0, t + delta)
        return x_next, t_next

    def critic_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        B = batch_actions.shape[0]
        rng, tc_rng, z_rng, zn_rng, d_rng = jax.random.split(rng, 5)
        valid = batch['valid'][..., -1]
        obs = batch['observations']
        next_obs = batch['next_observations'][..., -1, :]

        # ---- 1) environment Bellman target (noise-marginalized, no policy rollout) ----
        z_next = self._sample_v0_noise(zn_rng, batch_actions.shape)
        next_vs = jax.vmap(lambda zz: self._agg(
            self._v('target_critic', next_obs, zz, 0.0)))(z_next)
        next_v = next_vs.mean(axis=0)

        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_v




        # ---- 2) flow-TD transition ----
        t_c = jax.random.uniform(tc_rng, (B, 1))
        z = jax.random.normal(z_rng, batch_actions.shape)
        x_tc = t_c * batch_actions + (1.0 - t_c) * z

        vel = jax.lax.stop_gradient(self.network.select('actor_bc_flow')(obs, x_tc, t_c))
        delta, grounded = self._sample_step(d_rng, t_c)
        x_next, t_next = self._flow_step(x_tc, t_c, vel, delta, grounded)

        # ---- 3) fuse the three critic queries along the batch axis ----
        #   [a @ 1]            -> env Bellman target (grad)
        #   [x_tc @ t_c]       -> flow-TD prediction (grad)
        #   [x_next @ t_next]  -> flow-TD target    (stop-grad)
        # On boundary hits t_next = 1, so the third query reads the t=1 slice.
        obs3 = jnp.concatenate([obs, obs, obs], axis=0)
        xt3 = jnp.concatenate([
            self._with_time(batch_actions, 1.0),
            self._with_time(x_tc, t_c),
            self._with_time(x_next, t_next),
        ], axis=0)
        q_all = self.network.select('critic')(obs3, actions=xt3, params=grad_params)
        q1, q_tube, td_raw = jnp.split(q_all, 3, axis=1)

        clean_loss = (jnp.square(q1 - target_q) * valid).mean()
        td_target = self._agg(jax.lax.stop_gradient(td_raw))
        tube_loss = (jnp.square(q_tube - td_target[None]) * valid).mean()

        critic_loss = self.config["beta"]*clean_loss + tube_loss

        info = {'critic_loss': critic_loss,
                'clean_loss': clean_loss, 'tube_loss': tube_loss,
                'grounded_frac': grounded.mean(),
                'delta_mean': delta.mean(), 't_c_mean': t_c.mean(),
                'q_mean': q1.mean(), 'q_max': q1.max(), 'q_min': q1.min(),
                'q_tube_mean': q_tube.mean(),
                'td_target_mean': td_target.mean(),
                'next_v0_mean': next_v.mean()}
        return critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng, d_rng = jax.random.split(rng, 4)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape((pred - vel) ** 2,
                            (batch_size, self.config["horizon_length"], self.config["action_dim"]))
                * batch["valid"][..., None])
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        ginfo = {}
        if self.config['q_guidance']:
            # same step law as the critic (distribution match: query only where it was trained)
            obs_a = batch['observations']
            delta, bound = self._sample_step(d_rng, t)
            g = x_t + pred * delta
            # clip like an executed action at the boundary (straight-through gradient)
            g_st_clip = g + jax.lax.stop_gradient(jnp.clip(g, -1, 1) - g)
            g_eff = jnp.where(bound, g_st_clip, g)
            t_g = jnp.where(bound, 1.0, t + delta)

            use_adv = self.config['guidance_norm'] == 'adv'
            if use_adv:
                # fuse: [g_eff @ t_g] (guidance) + [x_0 @ 0] (advantage baseline)
                v_all = self.network.select('critic')(
                    jnp.concatenate([obs_a, obs_a], axis=0),
                    actions=jnp.concatenate(
                        [self._with_time(g_eff, t_g), self._with_time(x_0, 0.0)], axis=0))
                qg_raw, v0_raw = jnp.split(v_all, 2, axis=1)
                qg = jnp.mean(qg_raw, axis=0)
                v0 = jnp.mean(v0_raw, axis=0)
                adv = qg - v0
                adv_scale = jax.lax.stop_gradient(jnp.abs(adv).mean())
                q_loss = -(adv / (adv_scale + 1e-6)).mean()
                ginfo = {'adv_mean': adv.mean(), 'adv_scale': adv_scale, 'v0_mean': v0.mean()}
            else:
                qg = jnp.mean(self._v('critic', obs_a, g_eff, t_g), axis=0)
                q_loss = -qg.mean()
            ginfo.update({'g_delta_mean': delta.mean(), 'g_boundary_frac': bound.mean()})
        else:
            q_loss = jnp.zeros(())

        actor_loss = self.config['lmbda'] * bc_flow_loss + q_loss
        info = {'actor_loss': actor_loss, 'bc_flow_loss': bc_flow_loss, 'q_loss': q_loss}
        info.update(ginfo)
        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, donate_argnums=0)
    def update(self, batch):
        return self._update(self, batch)

    @partial(jax.jit, donate_argnums=0)
    def batch_update(self, batch):
        def body(agent, sub_batch):
            new_agent, info = self._update(agent, sub_batch)
            return new_agent, info

        new_agent, infos = jax.lax.scan(body, self, batch, unroll=2)
        info = jax.tree_util.tree_map(lambda x: x.mean(axis=0), infos)
        return new_agent, info

    @jax.jit
    def sample_actions(self, observations, rng=None):
        action_dim = self.config['action_dim'] * (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        noises = jax.random.normal(
            rng,
            (*observations.shape[: -len(self.config['ob_dims'])], action_dim))
        actions = self.compute_flow_actions(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_flow_actions(self, observations, noises):
        # Inference-only discretization. Training is strictly F-independent,
        # so flow_steps can be changed freely after training (F-transfer).
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        full_actions_t = cls._with_time(full_actions, 1.0)

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()

        # unified: single time-conditioned critic (input = [x, time_features(t)])
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            critic=(critic_def, (ex_observations, full_actions_t)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions_t)),
        )
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = jax.tree_util.tree_map(jnp.copy, params['modules_critic'])
        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='via',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            num_qs=2,
            flow_steps=10,          # inference-only discretization (training is F-independent)
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=False,
            q_guidance=True,
            guidance_norm='adv',
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            lmbda=3.0,
            beta=0.2,
            margin=0.05,
            num_v0_samples=8
        )
    )
    return config
