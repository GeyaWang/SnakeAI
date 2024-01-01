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


class ReLU(ActivationFunction):
    @staticmethod
    def func(x: float) -> float:
        if x < 0:
            return 0
        else:
            return x

    @staticmethod
    def prime_func(x: float) -> float:
        if x < 0:
            return 0
        else:
            return 1
