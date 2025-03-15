from abc import ABC, abstractmethod
from dataclasses import dataclass

import grid_exp_oe.models.base


@dataclass
class AlgorithmRequirements:
    policy_type: grid_exp_oe.models.base.PolicyType


@dataclass
class AlgorithmHParams(ABC):
    @classmethod
    @abstractmethod
    def requirements(self) -> AlgorithmRequirements:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def algo_id() -> str:
        ...
