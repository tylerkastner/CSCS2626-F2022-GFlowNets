import matplotlib.pyplot as plt
import torch
import tqdm
import yaml
import munch
from gfn.src.gfn.envs import Canvas
from gfn.src.gfn.estimators import LogitPFEstimator, LogitPBEstimator, LogZEstimator
from gfn.src.gfn.losses import TBParametrization, TrajectoryBalance
from gfn.src.gfn.samplers import MultiBinaryActionsSampler, CanvasTrajectoriesSampler
from gfn.src.gfn.containers.replay_buffer import ReplayBuffer
from gfn.src.gfn.utils import trajectories_to_training_samples, validate
from backbones.mnist_simple_unet import RewardNet



def train_grid_gfn(config, gfn_parametrization=None, trajectories_sampler=None, reward_net=None, gt_trajectories=None, n_train_steps=20000, verbose=0):

    env = Canvas(n_denoising_steps=config.env.n_denoising_steps, canvas_size=config.env.size, reward_net=reward_net.to('cuda'), discrete_block_size=1)

    if gfn_parametrization is None:
        # nn_kwargs = {'enc_chs': (1, 64, 64), 'dec_chs': (64, 64), 'num_class': 2}
        # logit_PF = LogitPFEstimator(env=env, module_name='UNet', nn_kwargs=nn_kwargs)
        # logit_PB = LogitPBEstimator(env=env, module_name='UNet', subtract_exit_actions=False, nn_kwargs=nn_kwargs)

        logit_PF = LogitPFEstimator(env=env, module_name='Debug', in_chs=(1, config.env.size, config.env.size), num_class=2)
        logit_PB = LogitPBEstimator(env=env, module_name='Debug', subtract_exit_actions=False, in_chs=(1, config.env.size, config.env.size), num_class=2, torso=logit_PF.module.torso)
        logZ = LogZEstimator(torch.tensor(0.))

        parametrization = TBParametrization(logit_PF, logit_PB, logZ)

        actions_sampler = MultiBinaryActionsSampler(estimator=logit_PF, max_traj_length=config.env.n_denoising_steps)
        trajectories_sampler = CanvasTrajectoriesSampler(env=env, actions_sampler=actions_sampler)
    else:
        parametrization = gfn_parametrization
        trajectories_sampler = trajectories_sampler

    loss_fn = TrajectoryBalance(parametrization=parametrization, canvas_actions=True, use_discrete_action_sampler=False, max_traj_length=config.env.n_denoising_steps)

    visited_terminating_states = (env.States.from_batch_shape((0,)) if not config.experiment.resample_for_validation else None)
    if config.experiment.use_replay_buffer > 0:
        replay_buffer = ReplayBuffer(env, loss_fn, capacity=config.experiment.replay_buffer_size)
        if gt_trajectories is not None:
            replay_buffer.add(gt_trajectories)
    else:
        replay_buffer = None

    params = [
        {"params": [val for key, val in parametrization.parameters.items() if key != "logZ"], "lr": 0.001},
        {"params": [parametrization.parameters["logZ"]], "lr": 0.01}
    ]
    optimizer = torch.optim.Adam(params)
    pbar = tqdm.trange(n_train_steps)
    rewards = []
    logzs = []
    losses = []
    for i in pbar:
        trajectories = trajectories_sampler.sample(n_trajectories=config.experiment.batch_size, max_traj_length=config.env.n_denoising_steps)
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


        pbar.set_description("Loss: {:.3f}, Rewards: {:.3f}".format(loss.item(), trajectories.rewards.detach().cpu().numpy().mean()))
        rewards.append(trajectories.rewards.detach().cpu().numpy().mean().item())
        losses.append(loss.detach().cpu().numpy().item())
        logzs.append(parametrization.parameters['logZ'].detach().cpu().numpy().item())

        if i % config.experiment.validation_interval == 0:
            plt.imshow(trajectories.last_states.states_tensor[0][0].detach().cpu().numpy(), vmin=0.0, vmax=1.0)
            plt.colorbar()
            plt.show()
            plt.plot(rewards)
            plt.title('Reward')
            plt.show()

            plt.plot(logzs)
            plt.title('LogZ')
            plt.show()

            plt.plot(losses)
            plt.title('Loss')
            plt.show()


    return parametrization, trajectories_sampler

if __name__ == '__main__':
    torch.manual_seed(1)
    with open("canvas_config.yml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    config = munch.munchify(config)

    reward_net = RewardNet().to('cuda')
    gfn_parametrization, trajectories_sampler = train_grid_gfn(config, reward_net=reward_net)
