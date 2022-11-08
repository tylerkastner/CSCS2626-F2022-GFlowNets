import copy
import numpy as np
import random

class Get_Traj():
    def __init__(self, env, end_pt=None, reward=None, method='reward'):
        self.env = env
        self.horizon = self.env.horizon
        self.ndim = self.env.ndim
        self._state = np.int32([0] * self.ndim)
        __, _, self.trajectory_reward  = self.env.true_density()
        
        if method == 'end_point':
            self.end_pt_idx = [end_pt]
        elif method == 'reward':
            self.end_pt_idx = np.where(self.trajectory_reward == reward)[0]
    
    def find_trajectories(self):
        p_trajectories = {} # point trajectories

        for i in self.end_pt_idx:
            trajectories = []
            pt_index = [i//self.horizon, i%self.horizon]
            z = np.zeros([self.horizon * self.ndim], dtype=np.float32)
            for j in range(self.ndim):
                z[j*self.horizon+pt_index[j]] = 1.0

            # tuple(current_state, action, state_idx, done)
            trajectories = [[(z, self.ndim, pt_index, True)]]

            for _ in range(sum(list(pt_index))):
                traj2 = []
                for j in range(self.ndim):
                    for k in range(len(trajectories)):
                        temp = self.get_prev_traj(trajectories[k],j)
                        if temp:
                            traj2.append(temp)
                    
                trajectories = copy.deepcopy(traj2)

            p_trajectories.update({i:trajectories})
            
        return p_trajectories

    def get_prev_traj(self, traj:list[tuple[list,int]], action):
        curr_state = copy.deepcopy(traj[0][0])
        if curr_state[self.horizon*action]:
            return None
        s = np.where(curr_state==1.0)[0]
        idx  = s[(s>=action*self.horizon) & (s<(action+1)*self.horizon)][0]
        curr_state[idx], curr_state[idx-1] = 0.0, 1.0
        s[idx//self.horizon] = idx - 1
        return [(curr_state, action, (np.mod(s, self.horizon)).tolist(), False)] + traj

def generate_trajs(env, n=10000):
    horizon = env.horizon
    ndim = env.ndim
    __, _, trajectory_reward  = env.true_density()

    trajectory_reward = np.append(trajectory_reward,trajectory_reward[-1])

    dataset = np.zeros([n, horizon*ndim + 2 + horizon*ndim]) # state + action + reward + next_state
    action = [i for i in range(ndim+1)]
    for i in range(n):
        state = [random.randint(0,horizon-1) for _ in range(ndim)]
        a = copy.deepcopy(action)
        for idx,j in zip([*range(ndim)][::-1], state[::-1]):
            if j == horizon-1:
                a.pop(idx)
        a = random.choice(a)
        for j in range(ndim):
            dataset[i][state[j]+horizon*j] = 1.0
        dataset[i][horizon*ndim] = a
        if a == ndim:
            dataset[i][horizon*ndim+1] = trajectory_reward[sum([j*(horizon**idx) for idx,j in zip(range(ndim),state)])]
            dataset[i][horizon*ndim+2:] = dataset[i][:horizon*ndim]
        else:
            state[a] = state[a]+1
            for j in range(ndim):
                dataset[i][horizon*ndim + 2+state[j]+horizon*j] = 1.0
        
    np.savetxt("old/samples_trajs.csv", dataset, delimiter=",")