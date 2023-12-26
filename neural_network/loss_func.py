import numpy as np
from abc import ABC, abstractmethod


class LossFunc(ABC):
    @staticmethod
    @abstractmethod
    def func(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def prime_func(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class MeanSquaredError(LossFunc):
    @staticmethod
    def func(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def prime_func(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size
