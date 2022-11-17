import re
import copy
import yaml
import munch
import numpy as np
from numpy.random import choice
import random
import matplotlib.pyplot as plt
import torch
import dill as pickle

from utils import render_distribution

from gfn.src.gfn.envs import HyperGrid
from gfn.src.gfn.containers import States, Trajectories

class Get_Traj():
    def __init__(self, env):
        self.env = env
        self.horizon = self.env.height
        self.ndim = self.env.ndim
        self._state = np.int32([0] * self.ndim)

    def sample_trajectory(self, end_pt):
        state = self.env.reset(batch_shape=(1,))

        trajectory_states: List[StatesTensor] = [state.states_tensor]
        trajectory_actions: List[ActionsTensor] = []

        traj_done = False

        while not traj_done:
            current_state = trajectory_states[-1][0]

            possible_actions = []

            # If we are at the end state our only possible action is the `end` action
            if torch.equal(current_state, end_pt):
                traj_done = True

                # TK: TODO: change this if we want ndim > 2
                possible_actions = [2]

            # If we can go left
            if current_state[0] < end_pt[0]:
                possible_actions.append(0)

            # If we can go left
            if current_state[1] < end_pt[1]:
                possible_actions.append(1)

            action = torch.tensor(random.choice(possible_actions))
            trajectory_actions.append(torch.unsqueeze(action, 0))

            current_state = self.env.States(states_tensor=current_state)
            new_state = self.env.step(current_state, action)
            trajectory_states.append(torch.unsqueeze(new_state.states_tensor, 0))

        trajectory_states = torch.stack(trajectory_states, dim=0)
        trajectory_states = self.env.States(states_tensor=trajectory_states)
        trajectory_actions = torch.stack(trajectory_actions, dim=0)

        trajectory = Trajectories(
            env=self.env,
            states=trajectory_states,
            actions=trajectory_actions,
        )

        return trajectory

def generate_trajs(env, n=10000, filename='sample_trajs.pkl', boltzmann=True):
    all_states = env.build_grid()
    all_rewards = env.reward(all_states)

    get_traj = Get_Traj(env)

    if boltzmann:
        reward_dict = {}

        for row_idx, row_tensor in enumerate(torch.exp(all_rewards)):
            for col_idx, value in enumerate(row_tensor):
                reward_dict[(row_idx, col_idx)] = float(value.numpy())

        states = list(map(lambda tup: torch.tensor(tup), reward_dict.keys()))

        # Workaround since we can only sample from finite list using choice
        state_dict = { i: state for i, state in enumerate(states) }
        probabilities = np.fromiter(reward_dict.values(), dtype=float) / sum(reward_dict.values())

        endpoints = [state_dict[state_idx] for state_idx in choice(len(states), n, p=probabilities)]

    else:
        optimal_states = (all_rewards == torch.max(all_rewards)).nonzero()

        endpoints = [random.choice(optimal_states) for _ in range(n)]

    sample_trajs = []
    for i in range(n):
        endpoint = endpoints[i]

        traj = get_traj.sample_trajectory(end_pt=endpoint)
        sample_trajs.append(traj)

    return sample_trajs


    # TK: TODO: Deal with pickling - leave in ram for now

    # with open(filename, 'wb') as f:
    #     pickle.dump(sample_trajs, f)



if __name__ == '__main__':

    with open("config.yml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    config = munch.munchify(config)

    env = HyperGrid(ndim=config.env.ndim,
                    height=config.env.height,
                    R0=0.01)

    generate_trajs(env=env)
