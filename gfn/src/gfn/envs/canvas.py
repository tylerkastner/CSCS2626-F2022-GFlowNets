from typing import ClassVar, Literal, Tuple, cast

import torch
from einops import rearrange
from gymnasium.spaces import Discrete, MultiDiscrete, MultiBinary
from torchtyping import TensorType
from typing import Optional, Tuple, Union
from copy import deepcopy

import utils
from gfn.src.gfn.containers.states import TimeDependentStates
from gfn.src.gfn.envs.canvas_env import CanvasEnv


TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]



class Canvas(CanvasEnv):
    def __init__(self, n_denoising_steps: int, reward_net: object, canvas_channels: int = 1, canvas_size: int = 28,
        discrete_block_size: int = 4, n_time_features: int = 2, device_str: Literal["cpu", "cuda"] = "cuda"):
        self.canvas_channels = canvas_channels + n_time_features
        self.canvas_size = canvas_size
        self.reward_net = reward_net
        self.n_denoising_steps = n_denoising_steps
        self.dx = 1 / n_denoising_steps * discrete_block_size
        #self.exit_action = torch.zeros((canvas_channels, canvas_size, canvas_size)).to(device_str)
        self.n_time_features = n_time_features
        self.t = 0

        s0 = torch.zeros((self.canvas_channels, self.canvas_size, self.canvas_size), dtype=torch.float32, device=torch.device(device_str))
        sf = torch.full((self.canvas_channels, self.canvas_size, self.canvas_size), fill_value=-1.0, dtype=torch.float32, device=torch.device(device_str))

        action_space = MultiBinary([self.canvas_size, self.canvas_size])

        super().__init__(
            n_denoising_steps=n_denoising_steps,
            action_space=action_space,
            s0=s0,
            sf=sf,
            device_str=device_str,
            n_time_features=n_time_features
        )

    def make_States_class(self) -> type[TimeDependentStates]:
        "Creates a States class for this environment"
        env = self

        class CanvasStates(TimeDependentStates):

            state_shape: ClassVar[tuple[int, ...]] = (env.canvas_channels, env.canvas_size, env.canvas_size)
            s0 = env.s0
            sf = env.sf

            # Instantiate class variables as instance variables so we can demote instance of
            # CanvasStates to an instance of States
            def __init__(self, states_tensor: StatesTensor, forward_masks: ForwardMasksTensor | None = None, backward_masks: BackwardMasksTensor | None = None):
                self.state_shape = (env.canvas_channels, env.canvas_size, env.canvas_size)
                self.s0 = env.s0
                self.sf = env.sf
                super().__init__(states_tensor, forward_masks, backward_masks)

            @classmethod
            def make_random_states_tensor(cls, batch_shape: Tuple[int, ...]) -> StatesTensor:
                "Creates a batch of random states."
                states_tensor = torch.rand(0, 1, batch_shape + env.s0.shape, device=env.device)
                return states_tensor

            def make_masks(self) -> Tuple[ForwardMasksTensor, BackwardMasksTensor]:
                "Mask illegal (forward and backward) actions."
                forward_masks = torch.ones((*self.batch_shape, env.canvas_channels, env.canvas_size, env.canvas_size), dtype=torch.bool, device=env.device,)
                backward_masks = torch.ones((*self.batch_shape, env.canvas_channels, env.canvas_size, env.canvas_size), dtype=torch.bool, device=env.device,)

                return forward_masks, backward_masks

            def update_masks(self) -> None:
                "Update the masks based on the current states."
                # The following two lines are for typing only.
                self.forward_masks = cast(ForwardMasksTensor, self.forward_masks)
                self.backward_masks = cast(BackwardMasksTensor, self.backward_masks)

                if len(self.states_tensor.shape) == 4:
                    self.forward_masks = self.states_tensor[:, env.n_time_features:] + env.dx  < 1.0
                    self.backward_masks = self.states_tensor[:, env.n_time_features:] - env.dx  >= 0.0
                elif len(self.states_tensor.shape) == 5:
                    self.forward_masks = self.states_tensor[:, :, env.n_time_features:] + env.dx < 1.0
                    self.backward_masks = self.states_tensor[:, :, env.n_time_features:] - env.dx >= 0.0
                else:
                    raise NotImplementedError('Why is this happening')

            @classmethod
            def from_batch_shape(cls, batch_shape: tuple[int], random: bool = False):
                """Create a States object with the given batch shape, all initialized to s_0.
                If random is True, the states are initialized randomly. This requires that
                the environment implements the `make_random_states_tensor` class method.
                """
                if random:
                    states_tensor = cls.make_random_states_tensor(batch_shape)
                else:
                    states_tensor = cls.make_initial_states_tensor(batch_shape)
                return cls(states_tensor)

            @classmethod
            def make_initial_states_tensor(cls, batch_shape: tuple[int]) -> StatesTensor:
                state_ndim = len(cls.state_shape)
                assert cls.s0 is not None and state_ndim is not None
                return cls.s0.repeat(*batch_shape, *((1,) * state_ndim))

        return CanvasStates

    def is_exit_actions(self, actions: TensorLong, step: int, automatically_exit_on_final_step: bool) -> TensorBool:
        #is_exit_action_mask = torch.all(torch.all(actions == self.exit_action, dim=-1), dim=-1).squeeze()
        #exit_step_mask = torch.full_like(is_exit_action_mask, fill_value=step == self.n_denoising_steps)
        #return exit_step_mask | is_exit_action_mask if automatically_exit_on_final_step else is_exit_action_mask

        exit_step_mask = torch.full((actions.shape[0],), fill_value=step == self.n_denoising_steps, dtype=torch.bool, device=actions.device)
        return exit_step_mask

    def maskless_step(self, states: StatesTensor, actions: TensorLong) -> None:
        self.t += 1
        time_features = utils.frequency_features(self.t/self.n_denoising_steps, self.n_time_features)
        update = torch.zeros_like(states)
        update[:, :self.n_time_features] = time_features.to(update.device)[None,:,None,None]
        update[:, self.n_time_features:] = actions * self.dx
        states.add_(update)

    def maskless_backward_step(self, states: StatesTensor, actions: TensorLong) -> None:
        #states.add_( - actions * self.dx * self.discrete_block_size)
        raise NotImplementedError('Why is this called?')

    def reward(self, final_states: TimeDependentStates) -> TensorFloat:
        final_states_raw = final_states.states_tensor
        # reward = torch.exp(-self.reward_net(final_states_raw.to(torch.float32)).detach().squeeze(-1))
        reward = torch.exp((final_states_raw.mean(-1).mean(-1).mean(-1) + 1))
        return reward



    def reset(self, batch_shape: Union[int, Tuple[int]], random: bool = False) -> TimeDependentStates:
        "Instantiates a batch of initial states."
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        return self.States.from_batch_shape(batch_shape=batch_shape, random=random)

    def step(self, states: TimeDependentStates, actions: TensorLong, step: int) -> TimeDependentStates:
        """Function that takes a batch of states and actions and returns a batch of next
        states and a boolean tensor indicating sink states in the new batch."""
        new_states = deepcopy(states)
        valid_states: TensorBool = ~states.is_sink_state
        # valid_actions = actions[valid_states]


        # if new_states.forward_masks is not None:
            # new_forward_masks, _ = correct_cast(new_states.forward_masks, new_states.backward_masks)
            # valid_states_masks = new_forward_masks[valid_states]
            # valid_actions_bool = all(torch.gather(valid_states_masks, 1, valid_actions.unsqueeze(1)))
            # if not valid_actions_bool:
            #     raise NonValidActionsError("Actions are not valid")

        new_sink_states = self.is_exit_actions(actions, step, automatically_exit_on_final_step=False)
        new_states.states_tensor[new_sink_states] = self.sf
        new_sink_states = ~valid_states | new_sink_states

        not_done_states = new_states.states_tensor[~new_sink_states]
        not_done_actions = actions[~new_sink_states]

        self.maskless_step(not_done_states, not_done_actions)
        new_states.states_tensor[~new_sink_states] = not_done_states

        new_states.update_masks()
        return new_states


