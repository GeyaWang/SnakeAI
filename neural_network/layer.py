from .activation_func import ActivationFunction
from abc import ABC, abstractmethod
from enum import Enum, auto
import numpy as np
from typing import Type


class LayerType(Enum):
    FCL = auto()
    ACTIVATION_LAYER = auto()


class Layer(ABC):
    @abstractmethod
    def __init__(self):
        self.input = None  # save input from forward propagation for backward propagation

    @abstractmethod
    def forward(self, input_: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class FullyConnectedLayer(Layer):
    def __init__(self, learning_rate: float, input_size: int, output_size: int):
        super().__init__()

        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.random.rand(1, output_size) - 0.5

    def forward(self, input_: np.ndarray) -> np.ndarray:
        """Applies formula Y = B + XW"""

        try:
            self.input = input_
            output = self.biases + np.dot(self.input, self.weights)
            return output
        except ValueError:
            raise ValueError('Shapes of layers and inputs do not match')

    def backward(self, output_err_grad: np.ndarray) -> np.ndarray:
        """Use dE/dY to adjust weights and biases, outputs dE/dX"""

        # calculation
        weight_err_grad = np.dot(self.input.T, output_err_grad)  # dE/dW = X.T * dE/dY
        bias_err_grad = output_err_grad  # dE/dB = dE/dY
        input_err_grad = np.dot(output_err_grad, self.weights.T)  # dE/dX = dE/dY * W.T

        # adjust weights and biases
        self.weights -= self.learning_rate * weight_err_grad
        self.biases -= self.learning_rate * bias_err_grad

        return input_err_grad

    def __repr__(self) -> str:
        return f'FullyConnectedLayer(input_size={self.input_size}, output_size={self.output_size})'


class ActivationLayer(Layer):
    def __init__(self, activation_func: Type[ActivationFunction]):
        super().__init__()

        self.activation_func = activation_func

        self.vectorised_activation_func = np.vectorize(self.activation_func.func)
        self.vectorised_activation_prime_func = np.vectorize(self.activation_func.prime_func)

    def forward(self, input_: np.ndarray) -> np.ndarray:
        """Applies activation function to each input"""

        # note: vectorising a function is generally not efficient
        self.input = input_
        return self.vectorised_activation_func(self.input)

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """Applies formula dE/dX = dE/dY * f'(X)"""

        return np.multiply(output_grad, self.vectorised_activation_prime_func(self.input))

    def __repr__(self) -> str:
        return f'ActivationLayer(activation_func={self.activation_func}, activation_func_prime={self.vectorised_activation_prime_func})'
