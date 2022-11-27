# Learning Options via Compression (Love)
This repository contains the source code for "Learning Options via Compression"
presented at NeurIPS 2022.

[Yiding Jiang\*](https://yidingjiang.github.io/), [Evan Zheran Liu\*](https://cs.stanford.edu/~evanliu/), [Benjamin Eysenbach](https://ben-eysenbach.github.io/), [J. Zico Kolter](https://zicokolter.com/), [Chelsea Finn](https://ai.stanford.edu/~cbfinn/)

## Setup

This code requires Python3.7+. Install the dependencies in `requirements.txt`. We recommend using a `virtualenv`:

```
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Additionally, download the datafiles from this [Google Drive link](https://drive.google.com/file/d/1XhAxPoNOFKkbJ1pYMt8voxXkKqQhC-9w/view) and place them in the root `love/` directory. Then, extract them:

```
tar -xvf compile.tar.gz
tar -xvf world3d.tar.gz
tar -xvf checkpoints.tar.gz
```

## Hierarchical Reinforcement Learning Experiments

The hierarchical reinforcement learning experiments on the multi-task grid
world and 3D domain consist of two phases: 1) extracting skills from the
demonstrations and 2) using those skills to learn new tasks.
The first phase is implemented in `train_rl.py`, while the second phase is
implemented in `dqn/main.py`.
Below, we detail commands that reproduce the results in the paper for these two
phases respectively.

### Extracting learned skills

The following commands extract learned skills with LOVE on the grid world
domain and the 3D domain respectively:

```
PYOPENGL_PLATFORM=egl python train_rl.py \
    --name=grid_world_love \
    --coding_len_coeff=0.005 \
    --kl_coeff=0.0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --dataset-path=compile  \
    --max-iters=20000  \
    --state-size=8 \
    --use_min_length_boundary_mask \
    --latent-n=10
```

```
PYOPENGL_PLATFORM=egl python train_rl.py \
    --name=3d_love \
    --coding_len_coeff=0.001 \
    --kl_coeff=0.0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --dataset-path=miniworld  \
    --use_min_length_boundary_mask \
    --max-iters=20000 \
    --state-size=64 \
    --use_min_length_boundary_mask \
    --learn-rate=0.0001
```

To use VTA instead, set `---coding_len_coeff=0` and `--kl_coeff=0.05`.
To use DDO instead, set `---coding_len_coeff=0`, `--kl_coeff=1`, and add the
`--ddo` flag.
We also recommend changing the experiment name in the `--name` flag, though
this is optional.
The extracted skills are saved in the model checkpoints under
`experiments/{name}/model-{step}.ckpt`, where `{name}` is the argument provided to the `--name` flag.
We select the checkpoint with the best LOVE objective for LOVE and with the
best ELBO for the others, as described in the paper.
These objectives can be monitored via wandb by supplying the `--wandb` flag and
changing the `entity` argument in the `wandb.init` call in `train_rl.py` to
your own wandb username.

### Learning new tasks with the learned skills

To learn new tasks with the skills extracted above, invoke `dqn/main.py`
setting `-b checkpoint=\"{checkpoint_path}\"` to point at a model checkpoint
from above. For example, the following commands reproduce the results from the
paper with LOVE skills on the grid world and 3D domains respectively, using
released checkpoints.

First, set the `PYTHONPATH` to be the root of this directory:

```
export PYTHONPATH=/path/to/love
```

Then run one of these commands:

```
PYOPENGL_PLATFORM=egl python3 dqn/main.py love_grid_world -b agent.policy.epsilon_schedule.total_steps=500000 -b checkpoint=\"checkpoints/love_grid_world.ckpt\" -b threshold=0 -b sparse_reward=True -b visit_length=3 -b bc=False -b oracle=False --seed 0 -b env=\"compile\"
```

```
PYOPENGL_PLATFORM=egl python3 dqn/main.py love_3d -b agent.sync_target_freq=30000 -b agent.policy.epsilon_schedule.total_steps=250000 -b checkpoint=\"checkpoints/love_3d.ckpt\" -b threshold=0.05 -b sparse_reward=True -b visit_length=3 -b bc=False -b oracle=False -b recurrent=True --seed 0 -b env=\"3d\"
```

Note that these commands configure whether the reward is sparse or not with
`sparse_reward` and the number of objectis to pick up with `visit_length`.


## Sequence Segmentation Didactic Example

To run the didactic examples in the paper, run the following commands:

### Simple Colors
```
python train_mdl.py \
    --coding_len_coeff=0.1 \
    --use_abs_pos_kl=1.0 \
    --batch-size=512 \
    --seed=0 \
    --dataset-path=./data/toy_data.npy \
    --max-iters=30000 
```

### Conditional Colors
```
python train_mdl.py \
    --coding_len_coeff=0.1 \
    --use_abs_pos_kl=1.0 \
    --batch-size=512 \
    --seed=0 \
    --dataset-path=./data/toy_data_markov_3_option.npy \
    --max-iters=30000 
```

## Citation

If you use this code, please cite our paper.

```
@article{jiang2022learning,
  title={Learning Options via Compression},
  author={Jiang, Yiding and Liu, Evan Zheran and Eysenbach, Benjamin and Kolter, Zico and Finn, Chelsea},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```
