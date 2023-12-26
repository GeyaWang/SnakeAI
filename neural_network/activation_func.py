from math import tanh
from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    @staticmethod
    @abstractmethod
    def func(x: float) -> float:
        pass

    @staticmethod
    @abstractmethod
    def prime_func(x: float) -> float:
        pass


class Tanh(ActivationFunction):
    @staticmethod
    def func(x: float) -> float:
        return tanh(x)

    @staticmethod
    def prime_func(x: float) -> float:
        return 1 - tanh(x) ** 2
