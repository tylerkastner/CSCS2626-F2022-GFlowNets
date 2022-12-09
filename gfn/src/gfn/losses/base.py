import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch
from torchtyping import TensorType

from gfn.src.gfn.containers.states import States, correct_cast
from gfn.src.gfn.containers.trajectories import Trajectories
from gfn.src.gfn.containers.transitions import Transitions
from gfn.src.gfn.distributions import (
    EmpiricalTrajectoryDistribution,
    TrajectoryBasedTerminatingStateDistribution,
    TrajectoryDistribution,
)
from gfn.src.gfn.envs import Env
from gfn.src.gfn.estimators import LogitPBEstimator, LogitPFEstimator
from gfn.src.gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler

# Typing
LogPTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]


@dataclass
class Parametrization(ABC):
    """
    Abstract Base Class for Flow Parametrizations,
    as defined in Sec. 3 of GFlowNets Foundations.
    All attributes should be estimators, and should either have a GFNModule or attribute called `module`,
    or torch.Tensor attribute called `tensor` with requires_grad=True.
    """

    @abstractmethod
    def Pi(self, env: Env, n_samples: int, **kwargs) -> TrajectoryDistribution:
        pass

    def P_T(self, env: Env, n_samples: int, **kwargs) -> TrajectoryBasedTerminatingStateDistribution:
        return TrajectoryBasedTerminatingStateDistribution(self.Pi(env, n_samples, **kwargs))

    @property
    def parameters(self) -> dict:
        """
        Return a dictionary of all parameters of the parametrization.
        Note that there might be duplicate parameters (e.g. when two NNs share parameters),
        in which case the optimizer should take as input set(self.parameters.values()).
        """
        # TODO: use parameters of the fields instead, loop through them here
        parameters_dict = {}
        for estimator in self.__dict__.values():
            parameters_dict.update(estimator.named_parameters())

        return parameters_dict

    def save_state_dict(self, path: str):
        for name, estimator in self.__dict__.items():
            torch.save(estimator.named_parameters(), os.path.join(path, name + ".pt"))

    def load_state_dict(self, path: str):
        for name, estimator in self.__dict__.items():
            estimator.load_state_dict(torch.load(os.path.join(path, name + ".pt")))


@dataclass
class PFBasedParametrization(Parametrization, ABC):
    r"Base class for parametrizations that explicitly used $P_F$"
    logit_PF: LogitPFEstimator
    logit_PB: LogitPBEstimator

    def Pi(self, env: Env, n_samples: int = 1000, **actions_sampler_kwargs) -> TrajectoryDistribution:
        actions_sampler = DiscreteActionsSampler(self.logit_PF, **actions_sampler_kwargs)
        trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
        trajectories = trajectories_sampler.sample_trajectories(n_trajectories=n_samples)
        return EmpiricalTrajectoryDistribution(trajectories)


class Loss(ABC):
    "Abstract Base Class for all GFN Losses"

    def __init__(self, parametrization: Parametrization):
        self.parametrization = parametrization

    @abstractmethod
    def __call__(self, *args, **kwargs) -> TensorType[0, float]:
        pass


class EdgeDecomposableLoss(Loss, ABC):
    @abstractmethod
    def __call__(self, edges: Transitions) -> TensorType[0, float]:
        pass


class StateDecomposableLoss(Loss, ABC):
    @abstractmethod
    def __call__(self, states_tuple: Tuple[States, States]) -> TensorType[0, float]:
        """Unlike the GFlowNets Foundations paper, we allow more flexibility by passing a tuple of states,
        the first one being the internal states of the trajectories (i.e. non-terminal states), and the second one
        being the terminal states of the trajectories. If these two are not handled differently, then they should be
        concatenated together."""
        pass


class TrajectoryDecomposableLoss(Loss, ABC):
    def get_pfs_and_pbs(
        self,
        trajectories: Trajectories,
        fill_value: float = 0.0,
        temperature: float = 1.0,
        epsilon=0.0,
        no_pf: bool = False,
        canvas_actions: bool = False
    ) -> Tuple[LogPTrajectoriesTensor | None, LogPTrajectoriesTensor]:
        """Evaluate log_pf and log_pb for each action in each trajectory in the batch.
        This is useful when the policy used to sample the trajectories is different from the one used to evaluate the loss.

        Args:
            trajectories (Trajectories): Trajectories to evaluate.
            fill_value (float, optional): Value to use for invalid states (i.e. s_f that is added to shorter trajectories). Defaults to 0.0.

            The next parameters correspond to how the actions_sampler evaluates each action.
            temperature (float, optional): Temperature to use for the softmax. Defaults to 1.0.
            epsilon (float, optional): Epsilon to use for the softmax. Defaults to 0.0.
            no_pf (bool, optional): Whether to evaluate log_pf as well. Defaults to False.

        Raises:
            ValueError: if the trajectories are backward.

        Returns:
            Tuple[LogPTrajectoriesTensor | None, LogPTrajectoriesTensor]: A tuple of float tensors of shape (max_length, n_trajectories) containing the log_pf and log_pb for each action in each trajectory. The first one can be None.
        """
        # fill value is the value used for invalid states (sink state usually)
        if trajectories.is_backward:
            raise ValueError("Backward trajectories are not supported")

        valid_states = trajectories.states[~trajectories.states.is_sink_state]
        if canvas_actions:
            time_tensor = torch.arange(trajectories.states.states_tensor.shape[0]).to('cuda')
            time_tensor = torch.stack([time_tensor] * trajectories.states.states_tensor.shape[1]).T
            valid_time_tensor = time_tensor[~trajectories.states.is_sink_state.cpu()]

            valid_actions = trajectories.actions[~torch.all(torch.all(torch.all((trajectories.actions == -1), dim=-1), dim=-1), dim=-1)]
            if valid_states.batch_shape[0] != valid_actions.shape[0]:
                raise AssertionError("Something wrong happening with log_pf evaluations")

            valid_states.forward_masks, valid_states.backward_masks = correct_cast(valid_states.forward_masks, valid_states.backward_masks)

            valid_pf_logits = self.actions_sampler.get_logits(valid_states, valid_time_tensor)
            valid_pb_logits = self.backward_actions_sampler.get_logits(valid_states[~valid_states.is_initial_state], valid_time_tensor[~valid_states.is_initial_state])


            valid_log_pf_all = valid_pf_logits.log_softmax(dim=1)
            valid_log_pf_actions = torch.gather(valid_log_pf_all, dim=1, index=valid_actions)
            valid_log_pf_actions = valid_log_pf_actions.sum(-1).sum(-1).sum(-1)  # We can sum since its logs
            log_pf_trajectories = torch.full((trajectories.actions.shape[0], trajectories.actions.shape[1]), fill_value=fill_value, dtype=torch.float, device=trajectories.actions.device)
            log_pf_trajectories[~torch.all(torch.all(torch.all((trajectories.actions == -1), dim=-1), dim=-1), dim=-1)] = valid_log_pf_actions

            valid_log_pb_all = valid_pb_logits.log_softmax(dim=1)
            non_exit_valid_actions = valid_actions[~torch.all(torch.all(torch.all((valid_actions == 0), dim=-1), dim=-1), dim=-1).flatten()]
            valid_log_pb_actions = torch.gather(valid_log_pb_all, dim=1, index=non_exit_valid_actions)
            valid_log_pb_actions = valid_log_pb_actions.sum(-1).sum(-1).sum(-1)

            log_pb_trajectories = torch.full(size=(trajectories.actions.shape[0], trajectories.actions.shape[1]), fill_value=fill_value, dtype=torch.float, device=valid_actions.device)#torch.full_like(trajectories.actions, fill_value=fill_value,dtype=torch.float)
            log_pb_trajectories_slice = torch.full(size=(valid_actions.shape[0],), fill_value=fill_value, dtype=torch.float, device=valid_actions.device)
            log_pb_trajectories_slice[~torch.all(torch.all(torch.all((valid_actions == 0), dim=-1), dim=-1), dim=-1).flatten()] = valid_log_pb_actions
            log_pb_trajectories[~torch.all(torch.all(torch.all((trajectories.actions == -1), dim=-1), dim=-1), dim=-1)] = log_pb_trajectories_slice


        else:
            valid_actions = trajectories.actions[trajectories.actions != -1]

            # uncomment next line for debugging
            # assert trajectories.states.is_sink_state[:-1].equal(trajectories.actions == -1)

            if valid_states.batch_shape != tuple(valid_actions.shape):
                raise AssertionError("Something wrong happening with log_pf evaluations")

            valid_states.forward_masks, valid_states.backward_masks = correct_cast(valid_states.forward_masks, valid_states.backward_masks)
            log_pf_trajectories = None
            if not no_pf:
                valid_pf_logits = self.actions_sampler.get_logits(valid_states)
                valid_pf_logits = valid_pf_logits / temperature
                valid_log_pf_all = valid_pf_logits.log_softmax(dim=-1)
                valid_log_pf_all = (1 - epsilon) * valid_log_pf_all + \
                                   epsilon * valid_states.forward_masks.float() / valid_states.forward_masks.sum(dim=-1, keepdim=True)
                valid_log_pf_actions = torch.gather(valid_log_pf_all, dim=-1, index=valid_actions.unsqueeze(-1)).squeeze(-1)
                log_pf_trajectories = torch.full_like(trajectories.actions, fill_value=fill_value, dtype=torch.float)
                log_pf_trajectories[trajectories.actions != -1] = valid_log_pf_actions

            valid_pb_logits = self.backward_actions_sampler.get_logits(valid_states[~valid_states.is_initial_state])
            valid_log_pb_all = valid_pb_logits.log_softmax(dim=-1)
            non_exit_valid_actions = valid_actions[valid_actions != trajectories.env.n_actions - 1]
            valid_log_pb_actions = torch.gather(valid_log_pb_all, dim=-1, index=non_exit_valid_actions.unsqueeze(-1)).squeeze(-1)
            log_pb_trajectories = torch.full_like(trajectories.actions, fill_value=fill_value, dtype=torch.float)
            log_pb_trajectories_slice = torch.full_like(valid_actions, fill_value=fill_value, dtype=torch.float)
            log_pb_trajectories_slice[valid_actions != trajectories.env.n_actions - 1] = valid_log_pb_actions
            log_pb_trajectories[trajectories.actions != -1] = log_pb_trajectories_slice

        return log_pf_trajectories, log_pb_trajectories

    @abstractmethod
    def __call__(self, trajectories: Trajectories) -> TensorType[0, float]:
        pass
