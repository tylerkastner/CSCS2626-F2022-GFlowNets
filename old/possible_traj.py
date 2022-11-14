import re
import copy
import numpy as np
import random

class Get_Traj():
    def __init__(self, env, end_pt=45, reward=None, method='end_point'):
        self.env = env
        self.horizon = self.env.horizon
        self.ndim = self.env.ndim
        self._state = np.int32([0] * self.ndim)
        __, _, self.trajectory_reward  = self.env.true_density()
        
        # We want to find the end point or end points that generate same reward
        if method == 'end_point':
            self.end_pt_idx = [end_pt]
        elif method == 'reward':
            self.end_pt_idx = np.where(self.trajectory_reward == reward)[0]
    
    def find_trajectories(self):
        p_trajectories = {} # point trajectories

        for i in self.end_pt_idx:
            trajectories = []

            # Find its index of point, like 45 means the index of point (6,6) in (8,8) matrix
            pt_index = [i//self.horizon, i%self.horizon]

            # initialize end_state 
            z = np.zeros([self.horizon * self.ndim], dtype=np.int8)
            for j in range(self.ndim):
                z[j*self.horizon+pt_index[j]] = 1.0

            # tuple(current_state, action, state_idx, done)
            trajectories = [[(z, self.ndim, pt_index, True)]]

            # keep exploring current state's possible previous state and form possible complete 
            # trajectory
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
    trajs = []
    with open('old/complete_trajs.csv','r') as f:
        for row in f:
            trajs.append([int(s) for s in re.findall(r'\b\d+\b', row)])

    sample_trajs = random.sample(trajs, n)
    np.savetxt("old/sample_trajs.csv", np.asanyarray(sample_trajs,dtype=object),delimiter=",",fmt='%s')

    # horizon = env.horizon # 8
    # ndim = env.ndim # 2
    # # We get the ground trush reward from true density
    # __, _, trajectory_reward  = env.true_density()

    # # For len(trajectory)==horizon**ndim-1, missing last reward, We manually set the value of last 
    # # reward equal to its previous reward
    # trajectory_reward = np.append(trajectory_reward,trajectory_reward[-1])

    # # Based on the toy_grid_dag.py that represent its state in one hot encoding, we keep using that
    # # way. Eg. [0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0] 2nd and 11th are positive, standing for current 
    # # state[2,3] = state[2,11-horizon]
    # # We initialized a dataset in shape (n, state + action + reward + next_state) where state shape
    # # is (16,), action (1,), reward (1,)
    # dataset = np.zeros([n, horizon*ndim + 2 + horizon*ndim]) 

    # # Possible action is 0,1,2 if ndim == 2. 0 stands for moving in first dimension, [2,3] to [3,3], 
    # # 1 stands for moving in second dimension [2,3] to [2,4] and 2 stands for termination.
    # action = [i for i in range(ndim+1)]

    # # creating trajectory loop, randomly generating state and action, finding its reward and 
    # # calculating its next_state
    # for i in range(n):
    #     state = [random.randint(0,horizon-1) for _ in range(ndim)]
    #     a = copy.deepcopy(action)
    #     for idx,j in zip([*range(ndim)][::-1], state[::-1]):
    #         if j == horizon-1:
    #             a.pop(idx)
    #     a = random.choice(a)
    #     for j in range(ndim):
    #         dataset[i][state[j]+horizon*j] = 1.0
    #     dataset[i][horizon*ndim] = a
    #     if a == ndim:
    #         dataset[i][horizon*ndim+1] = trajectory_reward[sum([j*(horizon**idx) for idx,j in zip(range(ndim),state)])]
    #         dataset[i][horizon*ndim+2:] = dataset[i][:horizon*ndim]
    #     else:
    #         state[a] = state[a]+1
    #         for j in range(ndim):
    #             dataset[i][horizon*ndim + 2+state[j]+horizon*j] = 1.0

    # # save generated result into a csv file 
    # np.savetxt("old/samples_trajs.csv", dataset, delimiter=",")