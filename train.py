import torch
import numpy as np
import yaml
import munch

from sample_trajs import generate_trajs
from networks.reward_network import RewardNetwork
from gfn.src.gfn.envs import HyperGrid
from grid import train_grid_gfn
import matplotlib.pyplot as plt

def train(config, env):
  trajectories = generate_trajs(env=env, n=100)
  state_shape = env.ndim

  # # add 1 for action
  # reward_net = RewardNetwork(state_dim = state_shape + 1)

  reward_net = RewardNetwork(state_dim=state_shape)
  reward_optimizer = torch.optim.Adam(reward_net.parameters(), lr=1e-2)

  reward_losses = []

  n_gfn_sample = 100
  gfn_parametrization, trajectories_sampler_gfn = train_grid_gfn(config, None, reward_net=reward_net, n_train_steps=10)
  for i_traj, traj in enumerate(trajectories):
    # states, actions = traj.states.states_tensor, torch.unsqueeze(traj.actions, -1)
    # # Remove last state in trajectory since there is no reward to predict from it
    # states = states[:-1]
    #
    # states_and_actions = torch.cat((states, actions), dim=-1)
    # states_and_actions = states_and_actions.to(torch.float32)
    #
    # rewards = reward_net(states_and_actions)
    #
    # # No average since for now there is a single trajectory in batch - to change if we increase BS
    # trajectory_reward = torch.sum(rewards)
    #
    # # This is the Z which will be learnt by GFlowNet
    # Z = 1
    #
    # loss = trajectory_reward - np.log(Z)
    # reward_optimizer.zero_grad()
    # loss.backward()
    # reward_optimizer.step()
    #
    # reward_losses.append(loss.detach())

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    states = traj.states.states_tensor
    last_state = states[:-1].to(torch.float32)

    trajectory_reward = reward_net(last_state)

    # This is the Z which will be learnt by GFlowNet
    gfn_sample = trajectories_sampler_gfn.sample(n_gfn_sample).last_states.states_tensor.to(torch.float32)
    gfn_Z = torch.exp(gfn_parametrization.logZ.tensor)
    sample_likelihood = torch.exp(reward_net(gfn_sample)).detach() / gfn_Z.detach()
    Z = torch.mean(torch.exp(reward_net(gfn_sample)) / sample_likelihood)

    loss = trajectory_reward - torch.log(Z)
    loss = loss.mean()
    reward_optimizer.zero_grad()
    loss.backward()
    reward_optimizer.step()

    reward_losses.append(loss.detach())

    # Fit gfn to new reward function
    gfn_parametrization, trajectories_sampler_gfn = train_grid_gfn(config, gfn_parametrization, trajectories_sampler_gfn, reward_net=reward_net, n_train_steps=100)


if __name__ == '__main__':
  with open("config.yml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)
  config = munch.munchify(config)

  env = HyperGrid(ndim=config.env.ndim,
                  height=config.env.height,
                  R0=0.01)
  train(config, env)
