import torch
import numpy as np

from networks.reward_network import RewardNetwork
from old.dataloader import create_dataloader
from old.toy_grid_dag import GridEnv


def train(env=GridEnv):
  batch_size = 1
  loader = create_dataloader(batch_size)

  state_shape = env(horizon=8).observation_shape

  # add 1 for action
  reward_net = RewardNetwork(state_dim = state_shape + 1)
  reward_optimizer = torch.optim.Adam(reward_net.parameters(), lr=1e-2)

  reward_losses = []

  # This loop currently doesn't work with the stored trajectories, need to discuss
  for i_batch, batch in enumerate(loader):
    states, actions = batch

    rewards = reward_net(torch.cat((states, actions.reshape(-1, 1)), dim=-1)).detach().numpy()

    # No average since for now there is a single trajectory in batch - to change if we increase BS
    trajectory_reward = sum(rewards)

    # This is the Z which will be learnt by GFlowNet
    Z = 1

    loss = trajectory_reward - np.log(Z)
    reward_optimizer.zero_grad()
    loss.backward()
    reward_optimizer.step()

    reward_losses.append(loss.detach())


if __name__ == '__main__':
  train()
