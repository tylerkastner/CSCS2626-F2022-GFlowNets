import torch
import numpy as np
import yaml
import munch

from sample_trajs import generate_trajs
from networks.reward_network import RewardNetwork
from gfn.src.gfn.envs import HyperGrid

def train(env):
  trajectories = generate_trajs(env=env, n=100)

  state_shape = env.ndim

  # add 1 for action
  reward_net = RewardNetwork(state_dim = state_shape + 1)
  reward_optimizer = torch.optim.Adam(reward_net.parameters(), lr=1e-2)

  reward_losses = []

  for i_traj, traj in enumerate(trajectories):
    states, actions = traj.states.states_tensor, torch.unsqueeze(traj.actions, -1)
    # Remove last state in trajectory since there is no reward to predict from it
    states = states[:-1]

    states_and_actions = torch.cat((states, actions), dim=-1)
    states_and_actions = states_and_actions.to(torch.float32)

    rewards = reward_net(states_and_actions)

    # No average since for now there is a single trajectory in batch - to change if we increase BS
    trajectory_reward = torch.sum(rewards)

    # This is the Z which will be learnt by GFlowNet
    Z = 1

    loss = trajectory_reward - np.log(Z)
    reward_optimizer.zero_grad()
    loss.backward()
    reward_optimizer.step()

    reward_losses.append(loss.detach())


if __name__ == '__main__':
  with open("config.yml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)
  config = munch.munchify(config)

  env = HyperGrid(ndim=config.env.ndim,
                  height=config.env.height,
                  R0=0.01)
  train(env)
