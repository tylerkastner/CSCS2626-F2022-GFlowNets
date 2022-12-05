from typing import ClassVar, Literal, Tuple, cast

import torch
from einops import rearrange
from gymnasium.spaces import Discrete, MultiDiscrete, MultiBinary
from torchtyping import TensorType

from gfn.src.gfn.containers.states import States
from gfn.src.gfn.envs.canvas_env import CanvasEnv


TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]



class Canvas(CanvasEnv):
    def __init__(self, n_denoising_steps: int, reward_net: object, canvas_channels: int = 1, canvas_size: int = 28, device_str: Literal["cpu", "cuda"] = "cuda"):
        self.canvas_channels = canvas_channels
        self.canvas_size = canvas_size
        self.reward_net = reward_net
        self.n_denoising_steps = n_denoising_steps
        self.dx = 1 / n_denoising_steps
        self.exit_action = torch.zeros((canvas_channels, canvas_size, canvas_size)).to(device_str)

        s0 = torch.zeros((self.canvas_channels, self.canvas_size, self.canvas_size), dtype=torch.float32, device=torch.device(device_str))
        sf = torch.full((self.canvas_channels, self.canvas_size, self.canvas_size), fill_value=-1.0, dtype=torch.float32, device=torch.device(device_str))

        action_space = MultiBinary([self.canvas_size, self.canvas_size])

        super().__init__(
            n_denoising_steps=n_denoising_steps,
            action_space=action_space,
            s0=s0,
            sf=sf,
            device_str=device_str,
        )

    def make_States_class(self) -> type[States]:
        "Creates a States class for this environment"
        env = self

        class CanvasStates(States):

            state_shape: ClassVar[tuple[int, ...]] = (env.canvas_channels, env.canvas_size, env.canvas_size)
            s0 = env.s0
            sf = env.sf

            # Instantiate class variables as instance variables so we can demote instance of
            # CanvasStates to an instance of States
            def __init__(self, states_tensor: StatesTensor, forward_masks: ForwardMasksTensor | None = None, backward_masks: BackwardMasksTensor | None = None,):
                self.state_shape = (env.canvas_channels, env.canvas_size, env.canvas_size)
                self.s0 = env.s0
                self.sf = env.sf
                super().__init__(states_tensor, forward_masks, backward_masks,)

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

                self.forward_masks = self.states_tensor + env.dx <= 1.0
                self.backward_masks = self.states_tensor <= 0.0

        return CanvasStates

    def is_exit_actions(self, actions: TensorLong) -> TensorBool:
        return torch.all(torch.all(actions == self.exit_action, dim=-1), dim=-1).squeeze()

    def maskless_step(self, states: StatesTensor, actions: TensorLong) -> None:
        states.add_(actions * self.dx)

    def maskless_backward_step(self, states: StatesTensor, actions: TensorLong) -> None:
        states.add_(actions * self.dx)

    def reward(self, final_states: States) -> TensorFloat:
        final_states_raw = final_states.states_tensor
        reward = torch.exp(-self.reward_net(final_states_raw.to(torch.float32)).detach().squeeze(-1))
        return reward

    def get_states_indices(self, states: States) -> TensorLong:
        # states_raw = states.states_tensor
        #
        # canonical_base = self.height ** torch.arange(self.ndim - 1, -1, -1, device=states_raw.device)
        # indices = (canonical_base * states_raw).sum(-1).long()
        # return indices
        raise NotImplementedError('Too many indices in the canvas environment')

    def get_terminating_states_indices(self, states: States) -> TensorLong:
        raise NotImplementedError('Too many indices in the canvas environment')

    @property
    def n_states(self) -> int:
        raise NotImplementedError('Too many states in canvas env')

    @property
    def n_terminating_states(self) -> int:
        raise NotImplementedError('Too many terminanting states in canvas env')

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        raise NotImplementedError('Unknown')

    @property
    def log_partition(self) -> float:
        # grid = self.build_grid()
        # rewards = self.reward(grid)
        # return rewards.sum().log().item()
        raise NotImplementedError('Needs to be estimated')

    def build_grid(self) -> States:
        # "Utility function to build the complete grid"
        # H = self.height
        # ndim = self.ndim
        # grid_shape = (H,) * ndim + (ndim,)  # (H, ..., H, ndim)
        # grid = torch.zeros(grid_shape, device=self.device)
        # for i in range(ndim):
        #     grid_i = torch.linspace(start=0, end=H - 1, steps=H)
        #     for _ in range(i):
        #         grid_i = grid_i.unsqueeze(1)
        #     grid[..., i] = grid_i
        #
        # rearrange_string = " ".join([f"n{i}" for i in range(1, ndim + 1)])
        # rearrange_string += " ndim -> "
        # rearrange_string += " ".join([f"n{i}" for i in range(ndim, 0, -1)])
        # rearrange_string += " ndim"
        # grid = rearrange(grid, rearrange_string).long()
        # return self.States(grid)
        raise NotImplementedError('Too many')

    @property
    def all_states(self) -> States:
        # grid = self.build_grid()
        # flat_grid = rearrange(grid.states_tensor, "... ndim -> (...) ndim")
        # return self.States(flat_grid)
        raise NotImplementedError('Too many')

    @property
    def terminating_states(self) -> States:
        # return self.all_states
        raise NotImplementedError('Too many')
