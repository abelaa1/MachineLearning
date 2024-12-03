# -*- coding: utf-8 -*-

import os
import warnings
import math

import gym
import pygame
from algorithms.rl import RL
from algorithms.planner import Planner
from examples.test_env import TestEnv
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


class FrozenLake:
    def __init__(self):
        self.env = gym.make('FrozenLake-v1', map_name="4x4", render_mode=None)


class Blackjack:
    def __init__(self):
        self._env = gym.make('Blackjack-v1', render_mode=None)
        # Explanation of convert_state_obs lambda:
        # def function(state, done):
        # 	if done:
		#         return -1
        #     else:
        #         if state[2]:
        #             int(f"{state[0]+6}{(state[1]-2)%10}")
        #         else:
        #             int(f"{state[0]-4}{(state[1]-2)%10}")
        self._convert_state_obs = lambda state, done: (
            -1 if done else int(f"{state[0] + 6}{(state[1] - 2) % 10}") if state[2] else int(
                f"{state[0] - 4}{(state[1] - 2) % 10}"))
        # Transitions and rewards matrix from: https://github.com/rhalbersma/gym-blackjack-v1
        current_dir = os.path.dirname(__file__)
        file_name = 'blackjack-envP'
        f = os.path.join(current_dir, file_name)
        try:
            self._P = pickle.load(open(f, "rb"))
        except IOError:
            print("Pickle load failed.  Check path", f)
        self._n_actions = self.env.action_space.n
        self._n_states = len(self._P)

    @property
    def n_actions(self):
        return self._n_actions

    @n_actions.setter
    def n_actions(self, n_actions):
        self._n_actions = n_actions

    @property
    def n_states(self):
        return self._n_states

    @n_states.setter
    def n_states(self, n_states):
        self._n_states = n_states

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, P):
        self._P = P

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        self._env = env

    @property
    def convert_state_obs(self):
        return self._convert_state_obs

    @convert_state_obs.setter
    def convert_state_obs(self, convert_state_obs):
        self._convert_state_obs = convert_state_obs

class Plots:
    @staticmethod
    def grid_world_policy_plot(data, label, save):
        if not math.modf(math.sqrt(len(data)))[0] == 0.0:
            warnings.warn("Grid map expected.  Check data length")
        else:
            plt.clf()
            data = np.around(np.array(data).reshape((4, 4)), 2)
            df = pd.DataFrame(data=data)
            my_colors = ((0.0, 0.0, 0.0, 1.0), (0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
            cmap = LinearSegmentedColormap.from_list('Custom', my_colors, len(my_colors))
            ax = sns.heatmap(df, cmap=cmap, linewidths=1.0)
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([.4, 1.1, 1.9, 2.6])
            colorbar.set_ticklabels(['Left', 'Down', 'Right', 'Up'])
            plt.title(label)
            plt.savefig(save)

    @staticmethod
    def grid_values_heat_map(data, label, save):
        if not math.modf(math.sqrt(len(data)))[0] == 0.0:
            warnings.warn("Grid map expected.  Check data length")
        else:
            plt.clf()
            data = np.around(np.array(data).reshape((4, 4)), 2)
            df = pd.DataFrame(data=data)
            sns.heatmap(df, annot=True).set_title(label)
            plt.savefig(save)

    @staticmethod
    def v_iters_plot(data, label,save):
        plt.clf()
        df = pd.DataFrame(data=data)
        df.columns = [label]
        sns.set_theme(style="whitegrid")
        title = label + " v Iterations"
        sns.lineplot(x=df.index, y=label, data=df).set_title(title)
        plt.xlabel("Iterations")
        plt.savefig(save)



if __name__ == "__main__":

    # frozen_lake = FrozenLake()

    # # VI/PI
    # V, V_track, pi = Planner(frozen_lake.env.P).value_iteration(n_iters=1000)
    # max_value_per_iter = np.amax(V_track, axis=1)
    # Plots.v_iters_plot(max_value_per_iter, "Max State Values", "images4/ValueFLiters.png")
    # Plots.grid_values_heat_map(V, "State Values", "images4/ValueFLheat.png")
    # n_states = frozen_lake.env.observation_space.n
    # new_pi = list(map(lambda x: pi(x), range(n_states)))
    # s = int(math.sqrt(n_states))
    # Plots.grid_world_policy_plot(np.array(new_pi), "Grid World Policy", "images4/ValueFLpolicy.png")


    # V, V_track, pi = Planner(frozen_lake.env.P).policy_iteration()
    # max_value_per_iter = np.amax(V_track, axis=1)
    # Plots.v_iters_plot(max_value_per_iter, "Max State Values", "images4/PolicyFLiters.png")
    # Plots.grid_values_heat_map(V, "State Values", "images4/PolicyFLheat.png")
    # n_states = frozen_lake.env.observation_space.n
    # new_pi = list(map(lambda x: pi(x), range(n_states)))
    # s = int(math.sqrt(n_states))
    # Plots.grid_world_policy_plot(np.array(new_pi), "Grid World Policy", "images4/PolicyFLpolicy.png")

    # # Q-learning
    # Q, V, pi, Q_track, pi_track = RL(frozen_lake.env).q_learning()
    # max_q_value_per_iter = np.amax(np.amax(Q_track, axis=2), axis=1)
    # Plots.v_iters_plot(max_q_value_per_iter, "Max Q-Values", "images4/QFLiters.png")
    # Plots.grid_values_heat_map(V, "State Values", "images4/QFLheat.png")
    # n_states = frozen_lake.env.observation_space.n
    # new_pi = list(map(lambda x: pi(x), range(n_states)))
    # s = int(math.sqrt(n_states))
    # Plots.grid_world_policy_plot(np.array(new_pi), "Grid World Policy", "images4/QFLpolicy.png")


    # test_scores = TestEnv.test_env(env=frozen_lake.env, render=True, user_input=False, pi=pi)


    # # --------------------------------------------------------------------------------------------------------------------


    blackjack = Blackjack()

    # VI/PI
    # V, V_track, pi = Planner(blackjack.P).value_iteration()
    # max_value_per_iter = np.amax(V_track, axis=1)
    # Plots.v_iters_plot(max_value_per_iter, "Max State Values", "images4/ValueBJiters.png")
    # plt.clf()
    # fig, axs = plt.subplots(2)
    # fig.suptitle('Black Jack Value Iterations')
    # axs[0].imshow([V]*2, cmap="plasma", aspect="auto")
    # axs[1].plot(V)
    # plt.savefig("images4/ValueBJheat.png")


    # V, V_track, pi = Planner(blackjack.P).policy_iteration()
    # max_value_per_iter = np.amax(V_track, axis=1)
    # Plots.v_iters_plot(max_value_per_iter, "Max State Values", "images4/PolicyBJiters.png")
    # plt.clf()
    # fig, axs = plt.subplots(2)
    # fig.suptitle('Black Jack Policy Iterations')
    # axs[0].imshow([V]*2, cmap="plasma", aspect="auto")
    # axs[1].plot(V)
    # plt.savefig("images4/PolicyBJheat.png")

    # Q-learning
    Q, V, pi, Q_track, pi_track = RL(blackjack.env).q_learning(blackjack.n_states, blackjack.n_actions, blackjack.convert_state_obs)
    max_q_value_per_iter = np.amax(np.amax(Q_track, axis=2), axis=1)
    Plots.v_iters_plot(max_q_value_per_iter, "Max Q-Values", "images4/QBJiters.png")
    plt.clf()
    fig, axs = plt.subplots(2)
    fig.suptitle('Black Jack Q Learner')
    axs[0].imshow([V]*2, cmap="plasma", aspect="auto")
    axs[1].plot(V)
    plt.savefig("images4/QBJheat.png")



    # test_scores = TestEnv.test_env(env=blackjack.env, render=True, pi=pi, user_input=False,
    #                                convert_state_obs=blackjack.convert_state_obs)

#----------------------------------------------Sources-------------------------------------------------------------------------
"""
-------------------------------------------Sources----------------------------------------------
Code
https://github.com/jlm429/bettermdptools/blob/master/readme.md
https://github.com/jlm429/bettermdptools/blob/master/examples/blackjack.py
https://github.com/jlm429/bettermdptools/blob/master/examples/frozen_lake.py
https://github.com/jlm429/bettermdptools/blob/master/examples/plots.py
https://github.com/Farama-Foundation/Gymnasium
https://gymnasium.farama.org/environments/toy_text/blackjack/
https://gymnasium.farama.org/environments/toy_text/frozen_lake/
https://towardsdatascience.com/reinforcement-learning-solving-blackjack-5e31a7fb371f
"""