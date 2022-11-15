import re
import copy
import yaml
import munch
import numpy as np
import random
import matplotlib.pyplot as plt

from gfn.envs import HyperGrid

class Get_Traj():
    def __init__(self, env):
        self.env = env
        self.horizon = self.env.height
        self.ndim = self.env.ndim
        self._state = np.int32([0] * self.ndim)

        # _, _, self.traj_rewards  = self.env.true_density()
        # self.traj_rewards = np.append(self.traj_rewards,self.traj_rewards[-1])     
    
    def find_trajectories(self, end_pt):
        # Find its index of point, like 45 means the index of point (6,6) in (8,8) matrix
        pt_index = [end_pt//self.horizon, end_pt%self.horizon]

        # initialize end_state 
        end_state = np.zeros([self.horizon * self.ndim], dtype=np.int8)
        for j in range(self.ndim):
            end_state[j*self.horizon+pt_index[j]] = 1

        # tuple(current_state, action, state_idx, done)
        trajectory = [(end_state, self.ndim, pt_index, True)]

        init_state = np.zeros([self.horizon * self.ndim], dtype=np.int8)
        for j in range(self.ndim):
            init_state[self.horizon**j-1*(1-j)] = 1 # specific for two dimension

        while not (trajectory[0][0] == init_state).all():
            action = random.randint(0,self.ndim-1)
            while trajectory[0][0][action*self.horizon] :
                action = random.randint(0,self.ndim-1)
            trajectory = self.get_prev_traj(trajectory,action)

        s_a = []
        for i in range(len(trajectory)):
            s_a.extend((np.where(trajectory[i][0]==1)[0]%self.horizon).tolist())
            s_a.append(trajectory[i][1])

        return s_a

    def get_prev_traj(self, traj:list[tuple[list,int]], action):
        curr_state = copy.deepcopy(traj[0][0])
        # if curr_state[self.horizon*action]:
        #     return None
        s = np.where(curr_state==1.0)[0]
        idx  = s[(s>=action*self.horizon) & (s<(action+1)*self.horizon)][0]
        curr_state[idx], curr_state[idx-1] = 0.0, 1.0
        s[idx//self.horizon] = idx - 1
        return [(curr_state, action, (np.mod(s, self.horizon)).tolist(), False)] + traj

def generate_trajs(env, n=10000, K=1):
    all_states = env.build_grid()
    all_rewards = env.reward(all_states)
    all_rewards_vec = all_rewards.reshape(1,env.height**env.ndim)[0]
    get_traj = Get_Traj(env)

    p = np.exp(all_rewards_vec*K)/np.exp(all_rewards_vec*K).sum()
    end_points = np.random.choice(env.height**env.ndim, n, p=p.detach().cpu().numpy())

    plt.matshow(np.array([np.sum(end_points==i) for i in range(env.height**env.ndim)]).\
                reshape(env.height, env.height).T)
    plt.show()

    sample_trajs = []
    for i in range(n):
        end_pt = end_points[i]
        p_traj = get_traj.find_trajectories(end_pt=end_pt)
        sample_trajs.append(p_traj)

    np.savetxt("sample_trajs.csv", np.asanyarray(sample_trajs,dtype=object),delimiter=",",fmt='%s')

if __name__ == '__main__':

    with open("config.yml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    config = munch.munchify(config)

    env = HyperGrid(ndim=config.env.ndim,
                    height=config.env.height,
                    R0=0.01)

    generate_trajs(env=env)