import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import munch
import tqdm
import random

import time

from torch.utils.data import TensorDataset, DataLoader

from sample_trajs import generate_trajs, load_pickled_trajectories
from networks.reward_network import RewardNetwork
from gfn.src.gfn.envs import HyperGrid
from utils import render_distribution
import matplotlib.pyplot as plt
import copy
import itertools


# TK: Hacky dataloader until we figure out batch size (works but not clean)
def iterate_trajs(dataset, batch_size):
    random.shuffle(dataset)
    return ((pos // batch_size, dataset[pos:pos + batch_size]) for pos in range(0, len(dataset), batch_size))

class PG(nn.Module):
  def __init__(self, ndim, n_actions):
    super().__init__()
    self.state_shape = (ndim,)
    self.n_actions = n_actions
    self.model = nn.Sequential(
      nn.Linear(in_features = ndim, out_features = 128),
      nn.ReLU(),
      nn.Linear(in_features = 128 , out_features = 64),
      nn.ReLU(),
      nn.Linear(in_features = 64 , out_features = self.n_actions)
    )

  def forward(self, x):
      x = x.to(torch.float32)
      logits = self.model(x)
      return logits

  def predict_probs(self, states):
      logits = self.model(states.to(torch.float32)).detach()
      probs = F.softmax(logits, dim = -1).numpy()

      return probs

  def generate_session(self, env, t_max=1000):
    states, traj_probs, actions = [], [], []
    s = env.reset(batch_shape=(1,))
    q_t = 1.0
    for t in range(t_max):
      action_probs = self.predict_probs(s.states_tensor)[0]

      if s.states_tensor[0].numpy()[0] == env.height - 1:
        action_probs[0] = 0

      if s.states_tensor[0].numpy()[1] == env.height - 1:
        action_probs[1] = 0

      # Renormalize
      total_action_probability = sum(action_probs)

      action_probs = [action_prob / total_action_probability for action_prob in action_probs]

      if np.isnan(np.array(action_probs)).any():
        action_probs = [0,0,1]

      a = torch.tensor([np.random.choice(self.n_actions,  p = action_probs)])

      try:
        new_s = env.step(s, a)
      except:
        print('INVALID ACTION')
        print(s.states_tensor)
        print(action_probs)

      q_t *= action_probs[a]

      states.append(s)
      traj_probs.append(q_t)
      actions.append(a)

      s = new_s

      if torch.equal(a, torch.tensor([env.ndim])):
        break

    return states, actions


def train(config, env):
  all_states = env.build_grid()

  states_filename = f'sample_trajs_states_{env.ndim}d.pkl'
  actions_filename = f'sample_trajs_actions_{env.ndim}d.pkl'

  trajectories = load_pickled_trajectories(env=env,
                                           states_filename=states_filename,
                                           actions_filename=actions_filename)

  last_states = torch.cat([traj.states.states_tensor[-2] for traj in trajectories])
  last_states_one_hot = torch.nn.functional.one_hot(last_states, num_classes=config.env.height)
  if env.ndim == 2:
    last_states_grid = (last_states_one_hot[:,0,:][:,:,None] * last_states_one_hot[:,1,:][:,None]).to(torch.float32)
    last_states_grid = last_states_grid.mean(0)
    # render_distribution(last_states_grid, env.height, env.ndim)

  elif env.ndim == 3:
    last_states_grid = (last_states_one_hot[:,0,:][:,:,None,None] *\
                        last_states_one_hot[:,1,:][:,None,:,None] *\
                        last_states_one_hot[:,2,:][:,None,None,:]).to(torch.float32)
    last_states_grid = last_states_grid.mean(0)


  reward_net = RewardNetwork(state_dim=env.ndim)
  policy = PG(env.ndim, env.ndim + 1)

  reward_net_checkpoint = copy.deepcopy(reward_net)

  reward_optimizer = torch.optim.Adam(reward_net.parameters(), lr=1e-3)
  policy_optimizer = torch.optim.Adam(policy.parameters(), 1e-3)

  reward_losses_per_epoch = []
  sample_trajectories = []

  print('\nStart training reward net')
  pbar = tqdm.trange(config.baseline.n_epochs_reward_fn)
  for epoch in pbar:
    reward_losses_per_batch = []
    for items in tqdm.tqdm(
      iterate_trajs(trajectories,
                    batch_size = config.baseline.batch_size_reward_fn),
                    total = len(trajectories) / config.baseline.batch_size_reward_fn,
                    position = 0,
                    leave = True):
      i_batch, batch = items
      last_states = torch.cat([traj.states.states_tensor[-2] for traj in batch])

      trajectory_reward = reward_net(last_states)

      generated_traj_batch = []
      generated_traj_batch_actions = []
      for _ in range(config.baseline.num_rollouts_per_epoch):
        policy_traj, actions = policy.generate_session(env)

        sample_trajectories.append(policy_traj)
        generated_traj_batch.append(policy_traj)
        generated_traj_batch_actions.append(actions)

      last_sampled_states = torch.cat([traj[-1].states_tensor for traj in sample_trajectories])
      sampled_rewards = -reward_net(last_sampled_states)
      sampled_log_Z = torch.log(torch.mean(torch.exp(sampled_rewards)))

      loss = trajectory_reward + sampled_log_Z
      loss = torch.mean(loss)
      reward_optimizer.zero_grad()
      loss.backward()
      reward_optimizer.step()

      print(f'rew_loss: {loss}')

      reward_losses_per_batch.append(loss.detach())

      policy_loss = 0
      for idx_sample_traj, sample_traj in enumerate(generated_traj_batch):
        traj_actions = generated_traj_batch_actions[idx_sample_traj]

        sampled_traj_reward = reward_net(sample_traj[-1].states_tensor)

        def get_action_probs(state):
          action_probs = policy.predict_probs(state.states_tensor[0])

          if state.states_tensor[0].numpy()[0] == env.height - 1:
            action_probs[0] = 0

          if state.states_tensor[0].numpy()[1] == env.height - 1:
            action_probs[1] = 0

          # Renormalize
          total_action_probability = np.sum(action_probs)

          action_probs = [action_prob / total_action_probability for action_prob in action_probs]

          return action_probs


        # probs = nn.functional.softmax(logits, -1)

        probs = torch.tensor([get_action_probs(state) for state in sample_traj])
        # print(probs)

        # log_probs = nn.functional.log_softmax(logits, -1)

        log_probs = torch.log(probs)

        def to_one_hot(x, ndims):
          x = torch.tensor(x).type(torch.LongTensor).view(-1, 1)
          x_one_hot = torch.zeros(
              x.size()[0], ndims).scatter_(1, x, 1)
          return x_one_hot


        log_probs_for_actions = torch.sum(
            log_probs * to_one_hot(traj_actions, config.env.nactions), dim=1)

        entropy = -torch.mean(torch.sum(probs * log_probs), dim = -1)


        # print(np.zeros(len(sample_traj) - 1))
        # print(sampled_traj_reward.detach().numpy())
        # print(sampled_traj_reward.detach().numpy()[0])

        # cumulative_returns = np.concatenate((np.zeros(len(sample_traj) - 1), sampled_traj_reward.detach().numpy()[0]))
        # cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

        # reward = torch.concatenate((torch.zeros(len(sample_traj) - 1), sampled_traj_reward[0]))
        # print(f'rewards: {rewards}')


        reward = sampled_traj_reward[0][0]
        print(f'reward: {reward}')

        cumulative_returns = torch.tensor([(0.99 ** (len(sample_traj) - i - 1)) * reward for i in range(len(sample_traj))], requires_grad=True)
        print(f'cumulative_returns: {cumulative_returns}')

        policy_loss -= torch.mean(log_probs_for_actions * cumulative_returns - entropy * 1e-2)

        print(f'entropy: {entropy}')
        print(f'sampled_traj_reward: {sampled_traj_reward}')

        print(f'actions: {actions}')
        print(f'log_probs_for_actions: {log_probs_for_actions}')
        print('====== end ========')

      policy_optimizer.zero_grad()
      policy_loss.backward()
      policy_optimizer.step()

      print(f'policy_loss: {policy_loss}')




    average_loss_per_epoch = torch.mean(torch.stack(reward_losses_per_batch))
    pbar.set_description('{}'.format(average_loss_per_epoch))
    reward_losses_per_epoch.append(average_loss_per_epoch)


  plt.plot(reward_losses_per_epoch)
  plt.show()

  predicted_reward = -reward_net(all_states.states_tensor).squeeze().cpu().detach().numpy()
  render_distribution(predicted_reward, env.height, env.ndim, 'learnt_reward')

if __name__ == '__main__':
  with open("config.yml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)
  config = munch.munchify(config)
  config.env.nactions = config.env.ndim + 1

  env = HyperGrid(ndim=config.env.ndim,
                  height=config.env.height,
                  R0=0.01)
  train(config, env)
