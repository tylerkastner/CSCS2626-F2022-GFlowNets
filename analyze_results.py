import torch
import os
import yaml
import munch
import json
import numpy as np

from utils import render_distribution


def main(config):
  path = f'results/hypergrid_estimates/{config.analysis.ndim}/{config.analysis.height}'

  predicted_reward = -torch.load(os.path.join(path, 'predicted_reward.pt')).numpy()

  if config.analysis.ndim == 2:
    predicted_reward = predicted_reward.reshape((config.analysis.height, config.analysis.height))
  elif config.analysis.ndim == 3:
   predicted_reward = predicted_reward.reshape((config.analysis.height, config.analysis.height, config.analysis.height))
  elif config.analysis.ndim == 4:
   predicted_reward = predicted_reward.reshape((config.analysis.height, config.analysis.height, config.analysis.height, config.analysis.height))
  elif config.analysis.ndim == 5:
   predicted_reward = predicted_reward.reshape((config.analysis.height, config.analysis.height, config.analysis.height, config.analysis.height, config.analysis.height))

  true_reward = torch.load(os.path.join(path, 'true_reward.pt')).numpy()

  render_distribution(predicted_reward, config.analysis.height, config.analysis.ndim, 't')
  render_distribution(true_reward, config.analysis.height, config.analysis.ndim, 't')

  nonzero_reward_indices = np.nonzero(true_reward - .01)

  # print(predicted_reward[nonzero_reward_indices])
  # print(true_reward[nonzero_reward_indices])

  # print(np.abs(predicted_reward[nonzero_reward_indices] - true_reward[nonzero_reward_indices]))

  approx_kl = np.mean(np.abs(predicted_reward[nonzero_reward_indices] - true_reward[nonzero_reward_indices]))

  print(f'dimension = {config.analysis.ndim}, approx_kl={approx_kl}')

  with open('results/hypergrid_estimates/hypergrid.json', 'r') as jsonfile:
    distance_object = json.load(jsonfile)

  distance_object[str(config.analysis.ndim)][str(config.analysis.height)] = str(approx_kl)

  print(distance_object)
  with open('results/hypergrid_estimates/hypergrid.json', 'w') as jsonfile:
    json.dump(distance_object, jsonfile)


if __name__ == '__main__':
  with open("config.yml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)
  config = munch.munchify(config)

  main(config)
