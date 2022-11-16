from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch
from simple_parsing import choice, subgroups
from simple_parsing.helpers import JsonSerializable

from gfn.src.gfn.envs import Env
from gfn.src.gfn.estimators import (
    LogEdgeFlowEstimator,
    LogitPBEstimator,
    LogitPFEstimator,
    LogStateFlowEstimator,
    LogZEstimator,
)
from gfn.src.gfn.losses import (
    DBParametrization,
    DetailedBalance,
    FlowMatching,
    FMParametrization,
    Loss,
    Parametrization,
    SubTBParametrization,
    SubTrajectoryBalance,
    TBParametrization,
    TrajectoryBalance,
)


@dataclass
class GFNModuleConfig(JsonSerializable):
    module_name: str = choice(
        "NeuralNet", "Tabular", "Uniform", "Zero", default="NeuralNet"
    )
    n_hidden_layers: int = 2
    hidden_dim: int = 256
    activation_fn: str = "relu"

    def __post_init__(self):
        self.nn_kwargs = self.__dict__.copy()


@dataclass
class BaseLossConfig(JsonSerializable, ABC):
    @abstractmethod
    def parse(self, env: Env, **kwargs) -> Tuple[Parametrization, Loss]:
        pass


@dataclass
class FMLossConfig(BaseLossConfig):
    logF_edge: GFNModuleConfig = GFNModuleConfig()
    alpha: float = 1.0

    def parse(
        self,
        env: Env,
    ) -> Tuple[Parametrization, Loss]:

        logF_edge = LogEdgeFlowEstimator(
            env=env,
            **self.logF_edge.nn_kwargs,
        )
        parametrization = FMParametrization(logF_edge)

        loss = FlowMatching(parametrization, alpha=self.alpha)

        return parametrization, loss


@dataclass
class PFBasedLossConfig(BaseLossConfig, ABC):
    logit_PF: GFNModuleConfig = GFNModuleConfig()
    logit_PB: GFNModuleConfig = GFNModuleConfig()
    tied: bool = True

    def get_estimators(
        self,
        env: Env,
    ) -> Tuple[LogitPFEstimator, LogitPBEstimator]:

        logit_PF = LogitPFEstimator(env=env, **self.logit_PF.nn_kwargs)
        logit_PB_kwargs = self.logit_PB.nn_kwargs
        if (
            self.tied
            and self.logit_PF.module_name
            and self.logit_PB.module_name == "NeuralNet"
        ):
            torso = logit_PF.module.torso
        else:
            torso = None
        logit_PB_kwargs["torso"] = torso
        logit_PB = LogitPBEstimator(env=env, **logit_PB_kwargs)

        return (logit_PF, logit_PB)


@dataclass
class StateFlowBasedLossConfig(PFBasedLossConfig, ABC):
    logF_state: GFNModuleConfig = GFNModuleConfig()

    def get_estimators(
        self,
        env: Env,
    ) -> Tuple[LogitPFEstimator, LogitPBEstimator, LogStateFlowEstimator]:

        logit_PF, logit_PB = super().get_estimators(env)
        logF_state_kwargs = self.logF_state.nn_kwargs
        if (
            self.tied
            and self.logit_PF.module_name == "NeuralNet"
            and self.logF_state.module_name == "NeuralNet"
        ):
            torso = logit_PF.module.torso
        else:
            torso = None
        logF_state_kwargs["torso"] = torso
        logF_state = LogStateFlowEstimator(env=env, **self.logF_state.nn_kwargs)

        return (logit_PF, logit_PB, logF_state)


@dataclass
class DBLossConfig(StateFlowBasedLossConfig):
    def parse(
        self,
        env: Env,
    ) -> Tuple[Parametrization, Loss]:
        logit_PF, logit_PB, logF_state = self.get_estimators(env)

        parametrization = DBParametrization(logit_PF, logit_PB, logF_state)
        loss = DetailedBalance(parametrization)
        return (parametrization, loss)


@dataclass
class SubTBLossConfig(StateFlowBasedLossConfig):
    weighing: str = choice(
        "equal",
        "equal_within",
        "geometric",
        "TB",
        "DB",
        "ModifiedDB",
        "geometric_within",
        default="geometric_within",
    )
    lamda: float = 0.9

    def parse(
        self,
        env: Env,
    ) -> Tuple[Parametrization, Loss]:
        logit_PF, logit_PB, logF_state = self.get_estimators(env)

        parametrization = SubTBParametrization(logit_PF, logit_PB, logF_state)
        loss = SubTrajectoryBalance(
            parametrization, weighing=self.weighing, lamda=self.lamda
        )
        return (parametrization, loss)


@dataclass
class TBLossConfig(PFBasedLossConfig):
    logZ_init: float = 0.0
    reward_clip_min: float = 1e-5

    def parse(
        self,
        env: Env,
    ) -> Tuple[Parametrization, Loss]:
        logit_PF, logit_PB = self.get_estimators(env)
        logZ_tensor = torch.tensor(self.logZ_init, dtype=torch.float)
        logZ = LogZEstimator(tensor=logZ_tensor)
        parametrization = TBParametrization(logit_PF, logit_PB, logZ)
        loss = TrajectoryBalance(parametrization, self.reward_clip_min)
        return (parametrization, loss)


@dataclass
class LossConfig(JsonSerializable):
    loss: BaseLossConfig = subgroups(
        {
            "FM": FMLossConfig,
            "DB": DBLossConfig,
            "TB": TBLossConfig,
            "SubTB": SubTBLossConfig,
        },
        default=TBLossConfig(),
    )

    def parse(self, env: Env) -> Tuple[Parametrization, Loss]:
        return self.loss.parse(env=env)
