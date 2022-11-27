import gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import PIL
import os
import pickle
import itertools

YELLOW = np.zeros(shape=(32, 32, 3), dtype=np.float32)
GREEN = np.zeros(shape=(32, 32, 3), dtype=np.float32)
BLUE =  np.zeros(shape=(32, 32, 3), dtype=np.float32)
YELLOW[:, :, 0] = 1.0
YELLOW[:, :, 1] = 1.0
GREEN[:, :, 1] = 1.0
BLUE[:, :, 2] = 1.0


class StateDependentEnv(gym.Env):

    def __init__(self, option_idx=0) -> None:
        super().__init__()
        state_seq_1 = np.stack([GREEN, BLUE, GREEN])
        state_seq_1_name = ["green", "blue", "green"]
        state_seq_2 = np.stack([BLUE, BLUE, GREEN])
        state_seq_2_name = ["blue", "blue", "green"]
        state_seq_3 = np.stack([GREEN, BLUE, YELLOW])
        state_seq_3_name = ["green", "blue", "yellow"]
        self.all_sequences = [state_seq_1, state_seq_2, state_seq_3]
        self.all_sequences_names = [state_seq_1_name, state_seq_2_name, state_seq_3_name]

        option_1 = {"yellow": 0, "green": 3, "blue": 1}
        option_2 = {"yellow": 1, "green": 0, "blue": 2}
        option_3 = {"yellow": 2, "green": 1, "blue": 3}
        self.option_list = [option_1, option_2, option_3]

        self.num_sequence = len(self.all_sequences)
        self.num_option = len(self.option_list)
        self.option_per_episode = 3
        self.seq_len = len(state_seq_1_name)
        self.total_length = self.seq_len * self.option_per_episode

        all_sequence_option_combination = list(itertools.product(*[range(self.num_option) for _ in range(self.option_per_episode)]))
        self.option_sequence = all_sequence_option_combination[option_idx]

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(32,32,3))

    @property
    def t_in_sequence(self):
        return self.t % self.seq_len

    @property
    def current_color(self):
        return self.all_sequences_names[self.curr_state_sequence][self.t_in_sequence]

    @property
    def correct_action(self):
        return self.option_list[self.option_sequence[self.curr_option_n]][self.current_color]

    def step(self, action):
        reward = 1 if self.correct_action == action else 0
        # Update state
        self.t += 1
        if self.t_in_sequence == 0:  # sampling a new subsequence
            self.curr_state_sequence = np.random.randint(self.num_sequence)
            self.curr_option_n += 1
        done = self.t == self.total_length
        return self.get_obs(), reward, done, {}

    def reset(self):
        self.curr_state_sequence = np.random.randint(self.num_sequence)
        self.t = 0
        self.curr_option_n = 0
        return self.get_obs()

    def get_obs(self):
        return self.all_sequences[self.curr_state_sequence][self.t_in_sequence]


if __name__ == "__main__":
    env = StateDependentEnv()
    s = env.reset()
    print(f'Initial state {s[0, 0]}')
    print(f'seq_len {env.seq_len}')
    done = False
    while not done:
        print('='*40)
        print(f'current color: {env.current_color}')
        print(f'curr sequence {env.curr_state_sequence}')
        print(f't {env.t}')
        print(f't_in_sequence {env.t_in_sequence}')
        a = np.random.randint(env.action_space.n)
        s, r, done, _ = env.step(a)
        print(f'action {a}')
        print(f'correct action {env.correct_action}')
        print(f'state {s[0, 0]}')
        print(f'reward {r}')
        print(f'done {done}')
        print(f'new color: {env.current_color}')
