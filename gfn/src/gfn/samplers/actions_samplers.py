from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch.distributions import Categorical
from torchtyping import TensorType

from gfn.src.gfn.containers.states import States, correct_cast
from gfn.src.gfn.estimators import LogEdgeFlowEstimator, LogitPBEstimator, LogitPFEstimator

# Typing
Tensor2D = TensorType["batch_size", "n_actions"]
Tensor2D2 = TensorType["batch_size", "n_steps"]
Tensor1D = TensorType["batch_size", torch.long]


class ActionsSampler(ABC):
    """
    Base class for action sampling methods.
    """

    @abstractmethod
    def sample(self, states: States, step: int = None) -> Tuple[Tensor1D, Tensor1D]:
        """
        Args:
            states (States): A batch of states.

        Returns:
            Tuple[Tensor[batch_size], Tensor[batch_size]]: A tuple of tensors containing the log probabilities of the sampled actions, and the sampled actions.
        """
        pass


class BackwardActionsSampler(ActionsSampler):
    """
    Base class for backward action sampling methods.
    """

    pass


class DiscreteActionsSampler:
    """
    For Discrete environments.
    """

    def __init__(self, estimator: LogitPFEstimator | LogEdgeFlowEstimator,
                 temperature: float = 1.0, sf_bias: float = 0.0, epsilon: float = 0.0,) -> None:
        """Implements a method that samples actions from any given batch of states.

        Args:
            temperature (float, optional): scalar to divide the logits by before softmax. Defaults to 1.0.
            sf_bias (float, optional): scalar to subtract from the exit action logit before dividing by temperature. Defaults to 0.0.
            epsilon (float, optional): with probability epsilon, a random action is chosen. Defaults to 0.0.
        """
        self.estimator = estimator
        self.temperature = temperature
        self.sf_bias = sf_bias
        self.epsilon = epsilon

    def get_raw_logits(self, states: States) -> Tensor2D:
        """
        This is before illegal actions are masked out and the exit action is biased.
        Should be used for Discrete action spaces only.

        Returns:
            Tensor2D: A 2D tensor of shape (batch_size, n_actions) containing the logits for each action in each state in the batch.
        """
        logits = self.estimator(states)
        return logits

    def get_logits(self, states: States) -> Tensor2D:
        """Transforms the raw logits by masking illegal actions.

        Raises:
            ValueError: if one of the resulting logits is NaN.

        Returns:
            Tensor2D: A 2D tensor of shape (batch_size, n_actions) containing the transformed logits.
        """
        logits = self.get_raw_logits(states)

        if torch.any(torch.all(torch.isnan(logits), 1)):
            raise ValueError("NaNs in estimator")
        states.forward_masks, _ = correct_cast(
            states.forward_masks, states.backward_masks
        )
        logits[~states.forward_masks] = -float("inf")
        return logits

    def get_probs(self, states: States,) -> Tensor2D:
        """
        Returns:
            The probabilities of each action in each state in the batch.
        """
        logits = self.get_logits(states)
        logits[..., -1] -= self.sf_bias
        probs = torch.softmax(logits / self.temperature, dim=-1)
        return probs

    def sample(self, states: States) -> Tuple[Tensor1D, Tensor1D]:
        probs = self.get_probs(states)
        states.forward_masks, _ = correct_cast(
            states.forward_masks, states.backward_masks
        )
        if self.epsilon > 0:
            uniform_dist = (
                states.forward_masks.float()
                / states.forward_masks.sum(dim=-1, keepdim=True).float()
            )
            probs = (1 - self.epsilon) * probs + self.epsilon * uniform_dist
        dist = Categorical(probs=probs)
        with torch.no_grad():
            actions = dist.sample()
        actions_log_probs = dist.log_prob(actions)

        return actions_log_probs, actions


class BackwardDiscreteActionsSampler(DiscreteActionsSampler, BackwardActionsSampler):
    """
    For sampling backward actions in discrete environments.
    """

    def __init__(
        self,
        estimator: LogitPBEstimator,
        temperature: float = 1.0,
        epsilon: float = 0.0,
    ) -> None:
        """s_f is not biased in the backward sampler."""
        super().__init__(
            estimator, temperature=temperature, sf_bias=0.0, epsilon=epsilon
        )

    def get_logits(self, states: States) -> Tensor2D:
        logits = self.get_raw_logits(states)
        if torch.any(torch.all(torch.isnan(logits), 1)):
            raise ValueError("NaNs in estimator")
        _, states.backward_masks = correct_cast(
            states.forward_masks, states.backward_masks
        )
        logits[~states.backward_masks] = -float("inf")
        return logits

    def get_probs(self, states: States) -> Tensor2D:
        logits = self.get_logits(states)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        # The following line is hack that works: when probs are nan, it means
        # that the state is already done (usually during backward sampling).
        # In which case, any action can be passed to the backward_step function
        # making the state stay at s_0
        probs = probs.nan_to_num(nan=1.0 / probs.shape[-1])
        return probs












class MultiBinaryActionsSampler:

    def __init__(self, estimator: LogitPFEstimator | LogEdgeFlowEstimator,
                 temperature: float = 1.0, sf_bias: float = 0.0, epsilon: float = 0.0, max_traj_length: int = None) -> None:
        """Implements a method that samples actions from any given batch of states.

        Args:
            temperature (float, optional): scalar to divide the logits by before softmax. Defaults to 1.0.
            sf_bias (float, optional): scalar to subtract from the exit action logit before dividing by temperature. Defaults to 0.0.
            epsilon (float, optional): with probability epsilon, a random action is chosen. Defaults to 0.0.
        """
        self.estimator = estimator
        self.temperature = temperature
        self.sf_bias = sf_bias
        self.epsilon = epsilon
        self.max_traj_length = max_traj_length

    def get_raw_logits(self, states: States) -> Tensor2D:
        """
        This is before illegal actions are masked out and the exit action is biased.
        Should be used for Discrete action spaces only.

        Returns:
            Tensor2D: A 2D tensor of shape (batch_size, n_actions) containing the logits for each action in each state in the batch.
        """
        logits = self.estimator(states)
        return logits

    def get_logits(self, states: States, step=None) -> Tensor2D:
        """Transforms the raw logits by masking illegal actions.

        Raises:
            ValueError: if one of the resulting logits is NaN.

        Returns:
            Tensor2D: A 2D tensor of shape (batch_size, n_actions) containing the transformed logits.
        """
        logits = self.get_raw_logits(states)

        if torch.any(torch.all(torch.isnan(logits), 1)):
            raise ValueError("NaNs in estimator")
        states.forward_masks, _ = correct_cast(states.forward_masks, states.backward_masks)

        # for two channel
        logits.swapaxes(1, 2).swapaxes(2, 3)[..., 1][~states.forward_masks[:,0]] = -float('inf')
        if type(step) is int:
            if step == self.max_traj_length:
                logits[:, -1] = -float('inf')
        else:
            logits[step == self.max_traj_length, -1] = -float('inf')

        return logits

    def get_probs(self, states: States, step=None) -> Tensor2D:
        """
        Returns:
            The probabilities of each action in each state in the batch.
        """
        logits = self.get_logits(states, step=step)
        # logits[..., -1] -= self.sf_bias
        probs = torch.softmax(logits / self.temperature, dim=1)
        # probs = torch.sigmoid(logits.swapaxes(1,2).swapaxes(2,3) / self.temperature).swapaxes(2, 3).swapaxes(1, 2)
        return probs

    def sample(self, states: States, step=None) -> Tuple[Tensor1D, Tensor1D]:
        probs = self.get_probs(states, step=step)
        states.forward_masks, _ = correct_cast(states.forward_masks, states.backward_masks)
        if self.epsilon > 0:
            uniform_dist = (
                states.forward_masks.float()
                / states.forward_masks.sum(dim=-1, keepdim=True).float()
            )
            probs = (1 - self.epsilon) * probs + self.epsilon * uniform_dist
        # dist = torch.distributions.binomial.Binomial(probs=probs)
        dist = torch.distributions.multinomial.Multinomial(probs=probs.swapaxes(1,2).swapaxes(2,3))
        with torch.no_grad():
            actions = dist.sample()
        actions_log_probs = dist.log_prob(actions)

        actions = actions.swapaxes(2, 3).swapaxes(1, 2)
        actions = torch.argmax(actions, dim=1, keepdim=True)
        return actions_log_probs, actions

    def evaluate_log_probs(self, states, actions, step=None):
        probs = self.get_probs(states, step=step)
        # dist = torch.distributions.binomial.Binomial(probs=probs)
        # actions_log_probs = dist.log_prob(actions)
        dist = torch.distributions.multinomial.Multinomial(probs=probs.swapaxes(1, 2).swapaxes(2, 3))
        actions_log_probs = dist.log_prob(torch.nn.functional.one_hot(actions.squeeze(), num_classes=2))
        return actions_log_probs





class MultiBinaryBackwardActionsSampler:
    """
    For sampling backward actions in discrete environments.
    """

    def __init__(self, estimator: LogitPBEstimator, temperature: float = 1.0, epsilon: float = 0.0, max_traj_length: int = None) -> None:
        """s_f is not biased in the backward sampler."""
        self.estimator = estimator
        self.temperature = temperature
        self.epsilon = epsilon
        self.max_traj_length = max_traj_length

    def get_raw_logits(self, states: States) -> Tensor2D:
        """
        This is before illegal actions are masked out and the exit action is biased.
        Should be used for Discrete action spaces only.

        Returns:
            Tensor2D: A 2D tensor of shape (batch_size, n_actions) containing the logits for each action in each state in the batch.
        """
        logits = self.estimator(states)
        return logits

    def get_logits(self, states: States, step=None) -> Tensor2D:
        logits = self.get_raw_logits(states)
        if torch.any(torch.all(torch.isnan(logits), 1)):
            raise ValueError("NaNs in estimator")
        _, states.backward_masks = correct_cast(states.forward_masks, states.backward_masks)

        # for two channel
        logits.swapaxes(1, 2).swapaxes(2, 3)[..., 1][~states.backward_masks[:,0]] = -float('inf')
        # if type(step) is int:
        #     if step == self.max_traj_length:
        #         logits[:, -1] = -float('inf')
        # else:
        #     logits[step == self.max_traj_length, -1] = -float('inf')

        return logits



    def get_probs(self, states: States, step=None) -> Tensor2D:
        logits = self.get_logits(states, step=step)
        probs = torch.softmax(logits / self.temperature, dim=1)
        # probs = torch.sigmoid(logits.swapaxes(1,2).swapaxes(2,3) / self.temperature).swapaxes(2, 3).swapaxes(1, 2)

        #probs = probs.nan_to_num(nan=1.0 / probs.shape[-1])
        return probs




