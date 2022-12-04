from abc import ABC
from typing import Literal, Optional

from torchtyping import TensorType

from gfn.src.gfn.containers import States
from gfn.src.gfn.envs import Env
from gfn.src.gfn.envs.preprocessors.base import EnumPreprocessor
from gfn.src.gfn.modules import GFNModule, NeuralNet, Tabular, Uniform, ZeroGFNModule
from backbones.mnist_simple_unet import UNet

# Typing
OutputTensor = TensorType["batch_shape", "output_dim", float]


class FunctionEstimator(ABC):
    def __init__(self, env: Env, module: Optional[GFNModule] = None, output_dim: Optional[int] = None,
                 module_name: Optional[Literal["NeuralNet", "Uniform", "Tabular", "Zero", "UNet"]] = None,
                 **nn_kwargs,
                 ) -> None:

        """Base class for function estimators. Either module or (module_name, output_dim) must be provided.

        Args:
            env (Env): the environment.
            module (Optional[GFNModule], optional): The module to use. Defaults to None.
            output_dim (Optional[int], optional): Used only if module is None. Defines the output dimension of the module. Defaults to None.
            module_name (Optional[Literal[NeuralNet, Uniform, Tabular, Zero]], optional): Used only if module is None. What module to use. Defaults to None.
            **nn_kwargs: Keyword arguments to pass to the module, if module_name is NeuralNet.
        """
        self.env = env
        assert module_name is not None
        if module is None and output_dim is not None:
            if module_name == "NeuralNet":
                assert len(env.preprocessor.output_shape) == 1
                input_dim = env.preprocessor.output_shape[0]
                module = NeuralNet(input_dim=input_dim, output_dim=output_dim, **nn_kwargs,)
            elif module_name == "Uniform":
                module = Uniform(output_dim=output_dim)
            elif module_name == "Zero":
                module = ZeroGFNModule(output_dim=output_dim)
            elif module_name == "Tabular":
                module = Tabular(n_states=env.n_states, output_dim=output_dim,)
            else:
                raise ValueError(f"Unknown module_name {module_name}")
        elif module_name == 'UNet':
            module = UNet(enc_chs=nn_kwargs['nn_kwargs']['enc_chs'], dec_chs=nn_kwargs['nn_kwargs']['dec_chs'], num_class=nn_kwargs['nn_kwargs']['num_class']).to('cuda:0')

        self.module = module
        if isinstance(self.module, Tabular):
            self.preprocessor = EnumPreprocessor(env.get_states_indices)
        else:
            self.preprocessor = env.preprocessor

    def __call__(self, states: States) -> OutputTensor:
        return self.module(self.preprocessor(states))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.module})"

    def named_parameters(self) -> dict:
        return dict(self.module.named_parameters())

    def load_state_dict(self, state_dict: dict):
        self.module.load_state_dict(state_dict)


class LogEdgeFlowEstimator(FunctionEstimator):
    def __init__(self, env: Env, module: Optional[GFNModule] = None,
                 module_name: Optional[Literal["NeuralNet", "Uniform", "Tabular", "Zero"]] = None,
                 **nn_kwargs,
                 ) -> None:

        if module is not None:
            assert module.output_dim == env.n_actions

        super().__init__(env, module=module, output_dim=env.n_actions, module_name=module_name, **nn_kwargs,)


class LogStateFlowEstimator(FunctionEstimator):
    def __init__(self, env: Env, module: Optional[GFNModule] = None,
                 module_name: Optional[Literal["NeuralNet", "Uniform", "Tabular", "Zero"]] = None,
                 **nn_kwargs,
                 ):

        if module is not None:
            assert module.output_dim == 1
        super().__init__(env, module=module, output_dim=1, module_name=module_name, **nn_kwargs)


class LogitPFEstimator(FunctionEstimator):
    def __init__(self, env: Env, module: Optional[GFNModule] = None,
                 module_name: Optional[Literal["NeuralNet", "Uniform", "Tabular", "Zero","UNet"]] = None,
                 **nn_kwargs,
                 ):

        if module is not None:
            assert module.output_dim == env.n_actions

        super().__init__(env, module=module, output_dim=env.n_actions, module_name=module_name, **nn_kwargs,)


class LogitPBEstimator(FunctionEstimator):
    def __init__(self, env: Env, module: Optional[GFNModule] = None,
                 module_name: Optional[Literal["NeuralNet", "Uniform", "Tabular", "Zero", "UNet"]] = None,
                 subtract_exit_actions: bool = True,
                 **nn_kwargs,
                 ):

        if subtract_exit_actions:
            if module is not None:
                assert module.output_dim == env.n_actions - 1
            super().__init__(env, module=module, output_dim=env.n_actions - 1, module_name=module_name, **nn_kwargs,)
        else:
            super().__init__(env, module=module, output_dim=env.n_actions, module_name=module_name, **nn_kwargs, )


class LogZEstimator:
    def __init__(self, tensor: TensorType[0, float]) -> None:
        self.tensor = tensor
        assert self.tensor.shape == ()
        self.tensor.requires_grad = True

    def __repr__(self) -> str:
        return str(self.tensor.item())

    def named_parameters(self) -> dict:
        return {"logZ": self.tensor}

    def load_state_dict(self, state_dict: dict):
        self.tensor = state_dict["logZ"]
