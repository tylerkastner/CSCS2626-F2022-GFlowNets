import torch
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
from grid import train_grid_gfn
from utils import render_distribution
import matplotlib.pyplot as plt
import copy
import itertools

tf = lambda x: torch.FloatTensor(x)

# TK: Hacky dataloader until we figure out batch size (works but not clean)
def iterate_trajs(dataset, batch_size):
    random.shuffle(dataset)
    return ((pos // batch_size, dataset[pos:pos + batch_size]) for pos in range(0, len(dataset), batch_size))


def train(config, env):
  all_states = env.build_grid()

  states_filename = 'sample_trajs_states_{}d.pkl'.format(env.ndim)
  actions_filename = 'sample_trajs_actions_{}d.pkl'.format(env.ndim)
  if config.experiment.force_generate_dataset:
    generate_trajs(env=env,
                   n=100000,
                   states_filename=states_filename,
                   actions_filename=actions_filename
                   )
    trajectories = load_pickled_trajectories(env,
                                             states_filename=states_filename,
                                             actions_filename=actions_filename
                                             )
                                             
  else:
    trajectories = load_pickled_trajectories(env=env,
                                             states_filename=states_filename,
                                             actions_filename=actions_filename
                                             )

  last_states = torch.cat([traj.states.states_tensor[-2] for traj in trajectories])
  last_states_one_hot = torch.nn.functional.one_hot(last_states, num_classes=config.env.height)
  if env.ndim == 2:
    last_states_grid = (last_states_one_hot[:,0,:][:,:,None] * last_states_one_hot[:,1,:][:,None]).to(torch.float32)
    last_states_grid = last_states_grid.mean(0)
    render_distribution(last_states_grid, env.height, env.ndim)

  elif env.ndim == 3:
    last_states_grid = (last_states_one_hot[:,0,:][:,:,None,None] *\
                        last_states_one_hot[:,1,:][:,None,:,None] *\
                        last_states_one_hot[:,2,:][:,None,None,:]).to(torch.float32)
    last_states_grid = last_states_grid.mean(0)
    render_distribution(env.reward(all_states), env.height, env.ndim, 'true_reward_3d_{}'.format(env.height))
    render_distribution(last_states_grid, env.height, env.ndim, 'dataset_trajs_samples_3d_{}'.format(env.height), iso=[1e-5,0.002])

  elif env.ndim >= 4:
    print("Higher Dimension Visiualization needs to be developed.")


  reward_net = RewardNetwork(state_dim=env.ndim)
  reward_net_checkpoint = copy.deepcopy(reward_net)

  reward_optimizer = torch.optim.Adam(reward_net.parameters(), lr=1e-3)

  reward_losses_per_epoch = []
  gfn_z_per_epoch = []
  gt_z_per_epoch = []

  if config.experiment.use_gfn_z:
    n_gfn_sample = 200
    print('Train gfn to initial distribution...')
    gfn_parametrization, trajectories_sampler_gfn = train_grid_gfn(config, None, reward_net=reward_net, n_train_steps=200)
  print('\nStart training reward net')
  pbar = tqdm.trange(config.experiment.n_epochs_reward_fn)
  for epoch in pbar:
    reward_losses_per_batch = []
    for items in tqdm.tqdm(iterate_trajs(trajectories, batch_size=config.experiment.batch_size_reward_fn), total=len(trajectories)/config.experiment.batch_size_reward_fn, position=0, leave=True):
      i_batch, batch = items
      last_states = torch.cat([traj.states.states_tensor[-2] for traj in batch])

      trajectory_reward = reward_net(last_states)

      if config.experiment.use_gfn_z:
        # This is the Z which will be learnt by GFlowNet
        gfn_sample = trajectories_sampler_gfn.sample(n_gfn_sample).last_states.states_tensor.to(torch.float32)
        gfn_Z = torch.exp(gfn_parametrization.logZ.tensor)
        if config.experiment.retrain_on_the_fly:
          sample_likelihood = torch.exp(-reward_net(gfn_sample)).detach() / gfn_Z.detach()
        else:
          sample_likelihood = torch.exp(-reward_net_checkpoint(gfn_sample)).detach() / gfn_Z.detach()
        Z = torch.mean(torch.exp(-reward_net(gfn_sample)) / sample_likelihood)
      else:
        all_rewards = reward_net(all_states.states_tensor.reshape(-1, config.env.ndim))
        Z = torch.sum(torch.exp(-all_rewards))


      loss = trajectory_reward + torch.log(Z)
      loss = torch.mean(loss)
      reward_optimizer.zero_grad()
      loss.backward()
      reward_optimizer.step()

      reward_losses_per_batch.append(loss.detach())

      if config.experiment.use_gfn_z and config.experiment.retrain_on_the_fly:
        # Fit gfn to new reward function
        gfn_parametrization, trajectories_sampler_gfn = train_grid_gfn(config, gfn_parametrization,
                                                                       trajectories_sampler_gfn, reward_net=reward_net,
                                                                       n_train_steps=5, verbose=1)

    average_loss_per_epoch = torch.mean(torch.stack(reward_losses_per_batch))
    pbar.set_description('{}'.format(average_loss_per_epoch))
    reward_losses_per_epoch.append(average_loss_per_epoch)

    if config.experiment.use_gfn_z and epoch % config.experiment.full_gfn_retrain == 0:
      print('Fully retrain gfn...')
      reward_net_checkpoint = copy.deepcopy(reward_net)
      gfn_parametrization, trajectories_sampler_gfn = train_grid_gfn(config, None, None, reward_net=reward_net, n_train_steps=1000)

      all_rewards = reward_net(all_states.states_tensor.reshape(-1, config.env.ndim))
      Z = torch.sum(torch.exp(-all_rewards))
      gfn_z_per_epoch.append(gfn_parametrization.logZ.tensor.detach().numpy().item())
      gt_z_per_epoch.append(torch.log(Z).detach().numpy().item())
      print('GT Z is {} and gfn Z is {}'.format(torch.log(Z), gfn_parametrization.logZ.tensor))


  plt.plot(reward_losses_per_epoch)
  plt.show()

  plt.plot(gfn_z_per_epoch, label='GFN log(Z)')
  plt.plot(gt_z_per_epoch, label='GT log(Z)')
  plt.legend()
  plt.show()

if __name__ == '__main__':
  with open("config.yml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)
  config = munch.munchify(config)
  config.env.nactions = config.env.ndim + 1

  env = HyperGrid(ndim=config.env.ndim,
                  height=config.env.height,
                  R0=0.01)
  train(config, env)
