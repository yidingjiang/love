import sys
import argparse
import pyglet
import math
from pyglet.window import key
from pyglet import clock
import numpy as np
import gym
import gym_miniworld
from world3d import MultiTask3DEnv


def pickup_object(env, entity, turn_threshold=0.9):
    delta = entity.pos - env.agent.pos
    delta_norm = np.linalg.norm(delta)
    delta_dir = delta / delta_norm

    obs, act = [], []

    while delta_norm >= 2.7 * env.agent.radius:
        if delta_dir.dot(env.agent.dir_vec) < turn_threshold:
            a = env.actions.turn_left
        else:
            a = env.actions.move_forward
        s_tp1, _, _, _ = env.step(a)
        act.append(a)
        obs.append(s_tp1)
        delta = entity.pos - env.agent.pos
        delta_norm = np.linalg.norm(delta)
        delta_dir = delta / delta_norm

    a = env.actions.pickup
    s_tp1, _, _, _ = env.step(a)
    act.append(a)
    obs.append(s_tp1)
    return obs, act


def generate_trajectory(env):
    s = env.reset()
    obs, act = [s], []
    for _ in range(3):
        color = gym_miniworld.entity.COLOR_NAMES[obs[-1][1]]
        ent = next(ent for ent in env.entities if ent.color == color)
        new_obs, new_act = pickup_object(env, ent)
        obs += new_obs
        act += new_act
    return obs, act


if __name__ == "main":
    env = gym.make('MiniWorld-Hallway-v0')
