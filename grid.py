import matplotlib.pyplot as plt
import torch
import tqdm
import yaml
import munch
from utils import render_distribution
from gfn.src.gfn.envs import HyperGrid
from gfn.src.gfn.estimators import LogitPFEstimator, LogitPBEstimator, LogZEstimator
from gfn.src.gfn.losses import TBParametrization, TrajectoryBalance
from gfn.src.gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler
from gfn.src.gfn.containers.replay_buffer import ReplayBuffer
from gfn.src.gfn.utils import trajectories_to_training_samples, validate




def train_grid_gfn(config, gfn_parametrization=None, trajectories_sampler=None, reward_net=None, gt_trajectories=None, n_train_steps=1000, verbose=0, file_name=''):

    env = HyperGrid(ndim=config.env.ndim, height=config.env.height, R0=0.01, reward_net=reward_net)  # Grid of size 8x8x8x8
    # all_states = env.build_grid()
    # all_rewards = env.reward(all_states)
    # if verbose == 0:
    #     render_distribution(all_rewards, config.env.height, config.env.ndim, 'true_reward_{}_{}d'.format(file_name,config.env.ndim))

    if gfn_parametrization is None:
        logit_PF = LogitPFEstimator(env=env, module_name='NeuralNet')
        logit_PB = LogitPBEstimator(env=env, module_name='NeuralNet', torso=logit_PF.module.torso)  # To share parameters between PF and PB
        logZ = LogZEstimator(torch.tensor(0.))

        parametrization = TBParametrization(logit_PF, logit_PB, logZ)

        actions_sampler = DiscreteActionsSampler(estimator=logit_PF)
        trajectories_sampler = TrajectoriesSampler(env=env, actions_sampler=actions_sampler)
    else:
        parametrization = gfn_parametrization
        trajectories_sampler = trajectories_sampler

    loss_fn = TrajectoryBalance(parametrization=parametrization)

    visited_terminating_states = (
        env.States.from_batch_shape((0,)) if not config.experiment.resample_for_validation else None
    )
    if config.experiment.use_replay_buffer:
        replay_buffer = ReplayBuffer(env, loss_fn, capacity=config.experiment.replay_buffer_size)
        if gt_trajectories is not None:
            replay_buffer.add(gt_trajectories)
    else:
        replay_buffer = None

    params = [
        {"params": [val for key, val in parametrization.parameters.items() if key != "logZ"], "lr": 0.001},
        {"params": [parametrization.parameters["logZ"]], "lr": 0.1}
    ]
    optimizer = torch.optim.Adam(params)

    states_visited = 0
    unique_visited_states = {}
    for i in tqdm.trange(n_train_steps, disable=False if verbose==0 else True, position=0, leave=True):
        trajectories = trajectories_sampler.sample(n_trajectories=config.experiment.batch_size)
        training_samples = trajectories_to_training_samples(trajectories, loss_fn)
        if replay_buffer is not None:
            replay_buffer.add(training_samples)
            training_objects = replay_buffer.sample(n_trajectories=config.experiment.batch_size)
        else:
            training_objects = training_samples

        optimizer.zero_grad()
        loss = loss_fn(training_objects)
        loss.backward()
        optimizer.step()

        if visited_terminating_states is not None:
            visited_terminating_states.extend(trajectories.last_states)

        states_visited += len(trajectories)
        unique_visited_states.update((t.numpy().tobytes(), 1) for t in trajectories.last_states.states_tensor)
        to_log = {"loss": loss.item(), "states_visited": states_visited, 'n_unique_states_visited': len(unique_visited_states)}

        if (i+1) % config.experiment.validation_interval == 0 and verbose == 0:
            validation_info, final_states_dist_pmf = validate(env, parametrization, config.experiment.n_validation_samples, visited_terminating_states, return_terminating_distribution=True)
            to_log.update(validation_info)
            tqdm.tqdm.write(f"Iteration: {i}: {to_log}")

            render_distribution(final_states_dist_pmf.reshape([config.env.height]*config.env.ndim) * torch.exp(parametrization.logZ.tensor.detach()), config.env.height, config.env.ndim,
                                'emp_reward_{}_{}d_{}it'.format(file_name, config.env.ndim, i+1))

    if config.experiment.kl_div_measure and n_train_steps > 500:
        validation_info, final_states_dist_pmf = validate(env, parametrization, config.experiment.n_validation_samples, visited_terminating_states, return_terminating_distribution=True)
        gfn_reward = final_states_dist_pmf.reshape([config.env.height]*config.env.ndim) * torch.exp(parametrization.logZ.tensor.detach())
        return parametrization, trajectories_sampler, gfn_reward.reshape(-1, 1)
    return parametrization, trajectories_sampler

if __name__ == '__main__':
    with open("config.yml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    config = munch.munchify(config)
    gfn_parametrization, trajectories_sampler = train_grid_gfn(config)
