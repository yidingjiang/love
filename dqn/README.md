# Minimal PyTorch DQN Example

Most of the DQN implementation lives in `dqn.py`.
This file is organized as follows:
- The user should just instantiate a `DQNAgent` object, which enables taking
  actions via `act`, as well as updating from sampling from the replay buffer
  and taking a normal DQN step via `update`.
  These methods use epsilon-greedy Q-Learning, and handle all the decaying
  epsilon, and target syncing details.
  To create a `DQNAgent`, the user should use the `from_config` method, which
  takes a `Config` object (defined in `config.py`), which is just a nested
  key-value store (example later).
  See `configs/default.json` for a basic config, which contains all of the
  arguments expected by the `DQNAgent`.
  Under the hood, this implementation is Dueling Double Deep Q-Learning.
- To create a `DQNAgent`, the user should specify a `StateEmbedder`, which embeds the state.
  The Q-Network is just a linear layer on top of the state embedding.
  To implement a state embedder object, just subclass the `Embedder` class in `embed.py`,
  which also contains an example `MiniWorldEmbedder`.
  Then, modify `DQNPolicy.from_config` to swap this custom embedder with the `MiniWorldEmbedder`.
- To train the `DQNAgent`, just roll-out the policy and pass the gathered
  experiences to the agent via the `update` function.
  There's a relatively small example in `main.py`: see the `run_episode`
  method, which calls the `update` function via the `experience_observers`
  argument and just rolls out a single episode.
  The episode is stored in `Experience` objects, which are just
  (s, a, r, s')-tuples, defined in `rl.py`.
  This file also shows how to create a `Config` object from the JSON file.

Other details:
- `replay.py` defines the replay buffer.
- `schedule.py` defines a simple linear decaying schedule, useful for the
  e-greedy schedule.
- `main.py` also sets up a few convenient things for running experiments.
  Each experiment has its own name, which is stored under the
  `experiments/name` directory, e.g., by running: `python3 main.py name`.
  The git hash and config are logged to this directory.
