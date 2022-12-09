from typing import List, Optional

import torch
from torchtyping import TensorType

from gfn.src.gfn.containers import States, Trajectories
from gfn.src.gfn.envs import Env, CanvasEnv
from gfn.src.gfn.samplers.actions_samplers import ActionsSampler, BackwardActionsSampler

# Typing
StatesTensor = TensorType["n_trajectories", "state_shape", torch.float]
ActionsTensor = TensorType["n_trajectories", torch.long]
LogProbsTensor = TensorType["n_trajectories", torch.float]
DonesTensor = TensorType["n_trajectories", torch.bool]


class TrajectoriesSampler:
    def __init__(
        self,
        env: Env,
        actions_sampler: ActionsSampler,
    ):
        """Sample complete trajectories, or completes trajectories from a given batch states, using actions_sampler.
        Args:
            env (Env): Environment to sample trajectories from.
            actions_sampler (ActionsSampler): Sampler of actions.
        """
        self.env = env
        self.actions_sampler = actions_sampler
        self.is_backward = isinstance(actions_sampler, BackwardActionsSampler)

    def sample_trajectories(self, states: Optional[States] = None, n_trajectories: Optional[int] = None, ) -> Trajectories:
        if states is None:
            assert (n_trajectories is not None), "Either states or n_trajectories should be specified"
            states = self.env.reset(batch_shape=(n_trajectories,))
        else:
            assert (len(states.batch_shape) == 1), "States should be a linear batch of states"
            n_trajectories = states.batch_shape[0]

        device = states.states_tensor.device

        dones = states.is_initial_state if self.is_backward else states.is_sink_state

        trajectories_states: List[StatesTensor] = [states.states_tensor]
        trajectories_actions: List[ActionsTensor] = []
        trajectories_logprobs: List[LogProbsTensor] = []
        trajectories_dones = torch.zeros(n_trajectories, dtype=torch.long, device=device)
        trajectories_rewards = torch.zeros(n_trajectories, dtype=torch.float, device=device)

        step = 0

        while not all(dones):
            actions = torch.full((n_trajectories,),fill_value=-1,dtype=torch.long,device=device,)
            log_probs = torch.full((n_trajectories,), fill_value=0, dtype=torch.float, device=device)
            actions_log_probs, valid_actions = self.actions_sampler.sample(states[~dones])
            actions[~dones] = valid_actions
            log_probs[~dones] = actions_log_probs
            trajectories_actions += [actions]
            trajectories_logprobs += [log_probs]

            if self.is_backward:
                new_states = self.env.backward_step(states, actions)
            else:
                new_states = self.env.step(states, actions)
            sink_states_mask = new_states.is_sink_state

            step += 1

            new_dones = (new_states.is_initial_state if self.is_backward else sink_states_mask) & ~dones
            trajectories_dones[new_dones & ~dones] = step
            trajectories_rewards[new_dones & ~dones] = self.env.reward(states[new_dones & ~dones])
            states = new_states
            dones = dones | new_dones

            trajectories_states += [states.states_tensor]

        trajectories_states = torch.stack(trajectories_states, dim=0).detach()
        trajectories_states = self.env.States(states_tensor=trajectories_states)
        trajectories_actions = torch.stack(trajectories_actions, dim=0).detach()
        trajectories_logprobs = torch.stack(trajectories_logprobs, dim=0).detach()

        trajectories = Trajectories(
            env=self.env,
            states=trajectories_states,
            actions=trajectories_actions,
            when_is_done=trajectories_dones,
            is_backward=self.is_backward,
            rewards=trajectories_rewards,
            log_probs=trajectories_logprobs,
        )

        return trajectories

    def sample(self, n_trajectories: int) -> Trajectories:
        return self.sample_trajectories(n_trajectories=n_trajectories)









########################################################################
########################################################################
########################################################################



class CanvasTrajectoriesSampler:
    def __init__(self, env: CanvasEnv, actions_sampler: ActionsSampler,):
        """Sample complete trajectories, or completes trajectories from a given batch states, using actions_sampler.

        Args:
            env (Env): Environment to sample trajectories from.
            actions_sampler (ActionsSampler): Sampler of actions.
        """
        self.env = env
        self.actions_sampler = actions_sampler
        self.is_backward = isinstance(actions_sampler, BackwardActionsSampler)

    def sample_trajectories(self, states: Optional[States] = None, n_trajectories: Optional[int] = None, max_traj_length: int = float('inf')) -> Trajectories:
        if states is None:
            assert (n_trajectories is not None), "Either states or n_trajectories should be specified"
            states = self.env.reset(batch_shape=(n_trajectories,))
        else:
            assert (len(states.batch_shape) == 1), "States should be a linear batch of states"
            n_trajectories = states.batch_shape[0]

        device = states.states_tensor.device
        dones = states.is_initial_state if self.is_backward else states.is_sink_state

        trajectories_states: List[StatesTensor] = [states.states_tensor]
        trajectories_actions: List[ActionsTensor] = []
        trajectories_logprobs: List[LogProbsTensor] = []
        trajectories_dones = torch.zeros(n_trajectories, dtype=torch.long, device=device)
        trajectories_rewards = torch.zeros(n_trajectories, dtype=torch.float, device=device)

        step = 0
        while not all(dones):
            actions = torch.full((n_trajectories, self.env.canvas_channels, self.env.canvas_size, self.env.canvas_size), fill_value=-1.0, dtype=torch.int64, device=device,)
            # log_probs = torch.full((n_trajectories, self.env.canvas_channels, self.env.canvas_size, self.env.canvas_size), fill_value=0.0, dtype=torch.float, device=device)
            log_probs = torch.full((n_trajectories, self.env.canvas_size, self.env.canvas_size), fill_value=0.0, dtype=torch.float, device=device)
            # if step < max_traj_length-1:
            #     actions_log_probs, valid_actions = self.actions_sampler.sample(states[~dones])
            #     actions[~dones] = valid_actions
            #     log_probs[~dones] = actions_log_probs
            #     trajectories_actions += [actions]
            #     trajectories_logprobs += [log_probs]
            # else:
            #     actions[~dones] = self.env.exit_action.to(dtype=torch.int64)#torch.zeros_like(states[~done])
            #     log_probs = self.actions_sampler.evaluate_log_probs(states[~dones], actions)
            #     trajectories_actions += [actions]
            #     trajectories_logprobs += [log_probs]
            actions_log_probs, valid_actions = self.actions_sampler.sample(states[~dones], step=step)
            actions[~dones] = valid_actions
            log_probs[~dones] = actions_log_probs
            trajectories_actions += [actions]
            trajectories_logprobs += [log_probs]

            step += 1

            if self.is_backward:
                new_states = self.env.backward_step(states, actions)
            else:
                new_states = self.env.step(states, actions, step)
            sink_states_mask = new_states.is_sink_state


            new_dones = (new_states.is_initial_state if self.is_backward else sink_states_mask ) & ~dones
            trajectories_dones[new_dones & ~dones] = step
            trajectories_rewards[new_dones & ~dones] = self.env.reward(states[new_dones & ~dones])
            states = new_states
            dones = dones | new_dones

            trajectories_states += [states.states_tensor]

        trajectories_states = torch.stack(trajectories_states, dim=0).detach()
        trajectories_states = self.env.States(states_tensor=trajectories_states)
        trajectories_actions = torch.stack(trajectories_actions, dim=0).detach()
        trajectories_logprobs = torch.stack(trajectories_logprobs, dim=0).detach()

        trajectories = Trajectories(env=self.env,
                                    states=trajectories_states,
                                    actions=trajectories_actions,
                                    when_is_done=trajectories_dones,
                                    is_backward=self.is_backward,
                                    rewards=trajectories_rewards,
                                    log_probs=trajectories_logprobs,
                                    )

        return trajectories

    def sample(self, n_trajectories: int, max_traj_length: int = float('inf')) -> Trajectories:
        return self.sample_trajectories(n_trajectories=n_trajectories, max_traj_length = max_traj_length)










