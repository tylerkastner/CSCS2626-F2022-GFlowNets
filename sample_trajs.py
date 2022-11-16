import re
import copy
import yaml
import munch
import numpy as np
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

        # _, _, self.traj_rewards  = self.env.true_density()
        # self.traj_rewards = np.append(self.traj_rewards,self.traj_rewards[-1])     
    
    """
    TK: Rewrote this below
    """
    # def find_trajectories(self, end_pt):
    #     # Find its index of point, like 45 means the index of point (6,6) in (8,8) matrix
    #     pt_index = [end_pt//self.horizon, end_pt%self.horizon]

    #     # initialize end_state
    #     end_state = np.zeros([self.horizon * self.ndim], dtype=np.int8)
    #     for j in range(self.ndim):
    #         end_state[j*self.horizon+pt_index[j]] = 1

    #     # tuple(current_state, action, state_idx, done)
    #     trajectory = [(end_state, self.ndim, pt_index, True)]

    #     init_state = np.zeros([self.horizon * self.ndim], dtype=np.int8)
    #     for j in range(self.ndim):
    #         init_state[self.horizon**j-1*(1-j)] = 1 # specific for two dimension

    #     while not (trajectory[0][0] == init_state).all():
    #         action = random.randint(0,self.ndim-1)
    #         while trajectory[0][0][action*self.horizon] :
    #             action = random.randint(0,self.ndim-1)
    #         trajectory = self.get_prev_traj(trajectory,action)

    #     s_a = []
    #     for i in range(len(trajectory)):
    #         s_a.extend((np.where(trajectory[i][0]==1)[0]%self.horizon).tolist())
    #         s_a.append(trajectory[i][1])

    #     return s_a

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


    # def get_prev_traj(self, traj:list[tuple[list,int]], action):
    #     curr_state = copy.deepcopy(traj[0][0])
    #     # if curr_state[self.horizon*action]:
    #     #     return None
    #     s = np.where(curr_state==1.0)[0]
    #     idx  = s[(s>=action*self.horizon) & (s<(action+1)*self.horizon)][0]
    #     curr_state[idx], curr_state[idx-1] = 0.0, 1.0
    #     s[idx//self.horizon] = idx - 1
    #     return [(curr_state, action, (np.mod(s, self.horizon)).tolist(), False)] + traj

def generate_trajs(env, n=10000, filename='sample_trajs.pkl'):
    all_states = env.build_grid()
    all_rewards = env.reward(all_states)

    get_traj = Get_Traj(env)

    # Hack to get argmax in 2d
    optimal_states = (all_rewards == torch.max(all_rewards)).nonzero()

    sample_trajs = []
    for i in range(n):
        endpoint = random.choice(optimal_states)

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
