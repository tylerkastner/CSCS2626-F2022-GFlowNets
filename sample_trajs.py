import re
import yaml
import tqdm
import munch
import numpy as np
from numpy.random import choice
import random
import torch
from torchtyping import TensorType
import pickle
from typing import List
import itertools

from gfn.src.gfn.envs import HyperGrid
from gfn.src.gfn.containers import States, Trajectories

ActionsTensor = TensorType["action", torch.float]

class Get_Traj():
    def __init__(self, env):
        self.env = env
        self.horizon = self.env.height
        self.ndim = self.env.ndim
        self._state = np.int32([0] * self.ndim)

    def sample_trajectory(self, end_pt):
        state = self.env.reset(batch_shape=(1,))

        trajectory_states: List[States.StatesTensor] = [state.states_tensor]
        trajectory_actions: List[ActionsTensor] = []

        traj_done = False

        while not traj_done:
            current_state = trajectory_states[-1][0]

            possible_actions = []

            # If we are at the end state our only possible action is the `end` action
            if torch.equal(current_state, end_pt):
                traj_done = True
                possible_actions = [self.ndim]

            # current_state do not reach the end state
            else:
                for i in range(self.ndim):
                    if current_state[i] < end_pt[i]:
                        possible_actions.append(i)

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

def generate_trajs(
    env,
    n=10000,
    states_filename='sample_trajs_states.pkl',
    actions_filename='sample_trajs_actions.pkl',
    boltzmann=True
) -> None:
    all_states = env.build_grid()
    all_rewards = env.reward(all_states)

    get_traj = Get_Traj(env)

    if boltzmann:
        reward_dict = {}

        keys_ranges = [range(env.height)] * env.ndim
        exp_rewards = torch.exp(all_rewards)
        for key in itertools.product(*keys_ranges):
            reward_dict[key] = float(exp_rewards[key].numpy())

        states = list(map(lambda tup: torch.tensor(tup), reward_dict.keys()))

        # Workaround since we can only sample from finite list using choice
        state_dict = { i: state for i, state in enumerate(states) }
        probabilities = np.fromiter(reward_dict.values(), dtype=float) / sum(reward_dict.values())

        endpoints = [state_dict[state_idx] for state_idx in choice(len(states), n, p=probabilities)]
    else:
        optimal_states = (all_rewards == torch.max(all_rewards)).nonzero()

        endpoints = [random.choice(optimal_states) for _ in range(n)]

    sample_trajs = []
    for i in tqdm.tqdm(range(n), ascii=True, desc='Trajectory Generating:'):
        endpoint = endpoints[i]

        traj = get_traj.sample_trajectory(end_pt=endpoint)
        traj.states.__class__ = States

        sample_trajs.append(traj)

    with open(states_filename, 'wb') as f:
        pickle.dump([traj.states for traj in sample_trajs], f)

    with open(actions_filename, 'wb') as f:
        pickle.dump([traj.actions for traj in sample_trajs], f)

def load_pickled_trajectories(env, states_filename='sample_trajs_states.pkl', actions_filename='sample_trajs_actions.pkl') -> List[Trajectories]:
    with open(states_filename, 'rb') as f:
        states_list = pickle.load(f)

        # Promote states to original class since we used States class for pickling
        for states in states_list:
            states.__class__ = env.make_States_class()

    with open(actions_filename, 'rb') as f:
        actions_list = pickle.load(f)

    def make_traj_object(states, actions) -> Trajectories:
        return Trajectories(
            env=env,
            states=states,
            actions=actions,
        )

    trajectories = [make_traj_object(states, actions) for states, actions in zip(states_list, actions_list)]

    return trajectories


# if __name__ == '__main__':

#     with open("config.yml", "r") as ymlfile:
#         config = yaml.safe_load(ymlfile)
#     config = munch.munchify(config)

#     env = HyperGrid(ndim=config.env.ndim,
#                     height=config.env.height,
#                     R0=0.01)

#     generate_trajs(env=env, n=2)

#     load_pickled_trajectories(env=env)
