# VIA

Official training code for **VIA** (offline RL with flow policies via flow-time
temporal differences). The critic is defined on the entire generative path of
the flow policy: the boundary is grounded by the environment Bellman update,
the interior is propagated by TD along flow time, and no ODE is integrated
anywhere in training.

## Installation

```bash
conda create -n via python=3.10 -y
conda activate via
pip install jax[cuda12] flax optax distrax ml_collections ogbench \
            gymnasium wandb tqdm imageio absl-py
```

Datasets are downloaded automatically by [OGBench](https://github.com/seohongpark/ogbench)
on first use.

## Usage

```bash
python main.py --env_name=cube-triple-play-singletask-task1-v0 \
  --horizon_length=5 --agent.action_chunking=True \
  --agent.lmbda=3 --agent.beta=15
```

- Default agent: `agents/via.py`. Default W&B project: `via`.
- Manipulation tasks: `--horizon_length=5 --agent.action_chunking=True`
- Locomotion (maze) tasks: `--horizon_length=1 --agent.action_chunking=False --agent.discount=0.995`
- Sparse-reward variants (scene, puzzle): add `--sparse`
- Results are written to `exp/<project>/<group>/<env>/sd.../` (CSV) and W&B.

## Hyperparameters

λ = `--agent.lmbda` (behavior-cloning weight), β = `--agent.beta` (Bellman loss weight).

| Task | λ | β | | Task | λ | β |
|---|---|---|---|---|---|---|
| Antmaze-large | 3 | 1 | | Puzzle-3x3 | 7 | 0.2 |
| Antmaze-giant | 3 | 0.2 | | Puzzle-4x4 | 3 | 30 |
| Humanoid-medium | 7 | 0.15 | | Cube-double | 7 | 15 |
| Humanoid-large | 7 | 0.2 | | Cube-triple | 3 | 15 |
| Scene | 7 | 0.4 | | Cube-quadruple | 5 | 0.15 |

All other hyperparameters use the defaults in `agents/via.py:get_config()`.

## Repository structure

```
main.py           # training / evaluation entry point
agents/via.py     # the VIA agent (single file)
utils/            # networks, datasets, checkpointing
envs/             # OGBench environment and dataset loaders
evaluation.py     # rollout evaluation
```

## Acknowledgements

The training harness, environment loaders, and network/dataset utilities are
adapted from the [FQL](https://github.com/seohongpark/fql) codebase (MIT
License). Benchmarks are from [OGBench](https://github.com/seohongpark/ogbench).


