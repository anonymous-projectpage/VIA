import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets

from utils.flax_utils import save_agent, save_actor
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np
import os
import imageio


jax.config.update('jax_compilation_cache_dir',
                  os.path.expanduser('~/.cache/jax_comp_cache'))
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1)
# jax.config.update('jax_default_matmul_precision', 'tensorfloat32')

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed (overwritten by the runner).')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 5000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 200000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_bool('save_actor', False, 'Save the actor (velocity field) at the end of the offline phase.')
flags.DEFINE_integer('start_training', 5000, 'Step at which online training starts.')

flags.DEFINE_integer('utd_ratio', 1, 'Update to data ratio.')

flags.DEFINE_float('discount', 0.99, 'Discount factor.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 1, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/via.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, 'Proportion of the dataset to use.')
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints.')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory.')

flags.DEFINE_integer('horizon_length', 4, 'Action chunking length.')
flags.DEFINE_bool('sparse', False, 'Make the task sparse reward.')
flags.DEFINE_string('wandb_project', 'via', 'Wandb project name.')
flags.DEFINE_string('task', 'default', 'Task id, set automatically from env_name.')
flags.DEFINE_string('tag', '', 'Wandb run name suffix.')
flags.DEFINE_bool('save_all_online_states', False, 'Save all trajectories to npz.')

flags.DEFINE_enum('run_mode', 'all_tasks', ['one_task', 'all_tasks'],
                  'one_task: run --env_name only. all_tasks: run task1..task5.')
flags.DEFINE_integer('num_seeds', 5, 'Number of random seeds to run.')
flags.DEFINE_integer('meta_seed', -1, 'Master seed used to draw the random seeds. -1 means fully random.')

UPDATES_PER_ITER = 1


class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)


def crossed(i, interval, stride):
    """True exactly once per `interval` steps, regardless of the loop stride.

    With a stride of K the loop index jumps by K, so `i % interval == 0` only
    fires every lcm(K, interval) steps. This checks whether the window
    (i - stride, i] contains an interval boundary instead.
    """
    return interval > 0 and (i // interval) > ((i - stride) // interval)


def parse_env_and_task(env_name: str):
    """Split an env name into its base name and task id.

    humanoidmaze-giant-navigate-singletask-task3-v0
      -> base = humanoidmaze-giant-navigate-singletask
      -> task = task3
    """
    if '-task' in env_name:
        base, rest = env_name.split('-task', 1)
        task_id = 'task' + rest.split('-v0')[0]
    else:
        base = env_name.replace('-v0', '')
        task_id = 'task0'
    return base, task_id


def main(_):
    config = FLAGS.agent
    exp_name = get_exp_name(FLAGS.seed)

    base_env, task_id = parse_env_and_task(FLAGS.env_name)
    FLAGS.task = task_id

    run = setup_wandb(project=FLAGS.wandb_project, group=task_id, name=f'{base_env}{FLAGS.tag}')

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    if FLAGS.ogbench_dataset_dir is not None:
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f'{FLAGS.ogbench_dataset_dir}/*.npz')) if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0

    discount = config.discount
    
    config['horizon_length'] = FLAGS.horizon_length

    def process_train_dataset(ds):
        """Apply dataset proportion and sparse-reward conversion."""
        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(**{k: v[:new_size] for k, v in ds.items()})

        if FLAGS.sparse:
            # Build a new dataset instead of mutating the frozen one.
            sparse_rewards = (ds['rewards'] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict['rewards'] = sparse_rewards
            ds = Dataset.create(**ds_dict)
            num_zero = (ds['rewards'] == 0.0).sum()
            print('number of zero rewards:', int(num_zero))

        return ds

    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    prefixes = ['eval', 'env']
    if FLAGS.offline_steps > 0:
        prefixes.append('offline_agent')
    if FLAGS.online_steps > 0:
        prefixes.append('online_agent')

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f'{prefix}.csv'))
                     for prefix in prefixes},
        wandb_logger=wandb,
    )

    offline_init_time = time.time()

    # Offline RL
    K = UPDATES_PER_ITER
    pbar = tqdm.tqdm(total=FLAGS.offline_steps)
    for i in range(K, FLAGS.offline_steps + 1, K):
        log_step += K

        if FLAGS.ogbench_dataset_dir is not None and crossed(i, FLAGS.dataset_replace_interval, K):
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f'Using new dataset: {dataset_paths[dataset_idx]}', flush=True)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
            )
            train_dataset = process_train_dataset(train_dataset)

        batch = train_dataset.sample_sequence(
            config['batch_size'] * K,
            sequence_length=FLAGS.horizon_length, discount=discount)
        batch = jax.tree.map(
            lambda x: x.reshape((K, config['batch_size']) + x.shape[1:]), batch)

        agent, offline_info = agent.batch_update(batch)
        pbar.update(K)

        is_last = (i + K) > FLAGS.offline_steps

        if crossed(i, FLAGS.log_interval, K):
            logger.log(offline_info, 'offline_agent', step=log_step)

        if crossed(i, FLAGS.save_interval, K):
            save_agent(agent, FLAGS.save_dir, log_step)

        if is_last or crossed(i, FLAGS.eval_interval, K):
            # The action chunk is executed fully during eval.
            eval_info, _, renders = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=example_batch['actions'].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, 'eval', step=log_step)

            if FLAGS.video_episodes > 0:
                os.makedirs(f'videos/{base_env}', exist_ok=True)
                for vid_idx, video in enumerate(renders):
                    path = f'videos/{base_env}/{task_id}_eval_ep_{log_step}_{vid_idx}.mp4'
                    imageio.mimsave(
                        path,
                        video,
                        fps=30 // FLAGS.video_frame_skip,
                    )
    pbar.close()

    if FLAGS.save_actor:
        save_actor(agent, FLAGS.save_dir, log_step)

    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )

    ob, _ = env.reset()

    action_queue = []
    action_dim = example_batch['actions'].shape[-1]

    # Online RL
    update_info = {}

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)

        # The action chunk is executed fully during online RL.
        if len(action_queue) == 0:
            action = agent.sample_actions(observations=ob, rng=key)
            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)

        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data['steps'].append(i)
            data['obs'].append(np.copy(next_ob))
            data['qpos'].append(np.copy(state['qpos']))
            data['qvel'].append(np.copy(state['qvel']))
            if 'button_states' in state:
                data['button_states'].append(np.copy(state['button_states']))

        env_info = {}
        for key, value in info.items():
            if key.startswith('distance'):
                env_info[key] = value
        if i % 5000 == 0:
            logger.log(env_info, 'env', step=log_step)

        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            # Adjust reward for D4RL antmaze.
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)

        if done:
            ob, _ = env.reset()
            action_queue = []
        else:
            ob = next_ob

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample_sequence(
                config['batch_size'] * FLAGS.utd_ratio,
                sequence_length=FLAGS.horizon_length, discount=discount)
            batch = jax.tree.map(lambda x: x.reshape((
                FLAGS.utd_ratio, config['batch_size']) + x.shape[1:]), batch)

            agent, update_info['online_agent'] = agent.batch_update(batch)

        if crossed(i, FLAGS.log_interval, 1):
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps or crossed(i, FLAGS.eval_interval, 1):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, 'eval', step=log_step)

        if crossed(i, FLAGS.save_interval, 1):
            save_agent(agent, FLAGS.save_dir, log_step)

    end_time = time.time()

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {
            'steps': np.array(data['steps']),
            'qpos': np.stack(data['qpos'], axis=0),
            'qvel': np.stack(data['qvel'], axis=0),
            'obs': np.stack(data['obs'], axis=0),
            'offline_time': online_init_time - offline_init_time,
            'online_time': end_time - online_init_time,
        }
        if len(data['button_states']) != 0:
            c_data['button_states'] = np.stack(data['button_states'], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, 'data.npz'), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)


def make_random_seeds(n):
    """Draw n random seeds up front so the whole sweep is reproducible via --meta_seed."""
    rs = random.Random(FLAGS.meta_seed if FLAGS.meta_seed >= 0 else None)
    return [rs.randint(0, 2 ** 31 - 1) for _ in range(n)]


def expand_env_names(base_env_name):
    """Expand a singletask env name into task1..task5 when run_mode is all_tasks."""
    if FLAGS.run_mode != 'all_tasks':
        return [base_env_name]

    if '-task' in base_env_name:
        base_env_name = base_env_name.split('-task', 1)[0] + '-v0'

    if base_env_name.endswith('-singletask-v0'):
        return [
            base_env_name.replace('-singletask-v0', f'-singletask-task{i}-v0')
            for i in range(1, 6)
        ]

    print(f'[warn] "{base_env_name}" cannot be expanded into tasks. Running it as a single env.', flush=True)
    return [base_env_name]


def run(_):
    base_env_name = FLAGS.env_name
    base_save_dir = FLAGS.save_dir  # main() overwrites save_dir, so keep the original.

    env_names = expand_env_names(base_env_name)
    seeds = make_random_seeds(FLAGS.num_seeds)

    print(f'\n[run_mode] {FLAGS.run_mode}')
    print(f'[envs]     {env_names}')
    print(f'[seeds]    {seeds}\n', flush=True)

    total = len(seeds) * len(env_names)
    count = 0
    for r, seed in enumerate(seeds):
        for env_name in env_names:
            count += 1
            FLAGS.save_dir = base_save_dir
            FLAGS.env_name = env_name
            FLAGS.seed = seed

            print(f'\n########## [{count}/{total}] env={env_name} | seed={seed} '
                  f'(run {r + 1}/{len(seeds)}) ##########\n', flush=True)
            try:
                main(None)
            finally:
                wandb.finish()
                jax.clear_caches()

    FLAGS.env_name = base_env_name
    FLAGS.save_dir = base_save_dir


if __name__ == '__main__':
    app.run(run)
