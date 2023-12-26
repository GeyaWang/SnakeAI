import numpy as np
from .layer import LayerType, FullyConnectedLayer, ActivationLayer
from .loss_func import LossFunc
from typing import Callable, Type
import os
import pickle


class Model:
    def __init__(self, learning_rate: float = None, loss_func: Type[LossFunc] = None, layers: list[LayerType] = None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            raise ValueError('Specify a learning rate')

        if loss_func is not None:
            self.loss_func = loss_func
        else:
            raise ValueError('Specify a loss function')

        if layers is not None:
            self.layers = layers
        else:
            self.layers = []

    def __str__(self):
        return f'\nNetwork Object:\nlearning_rate={self.learning_rate}\nloss_func={self.loss_func}\nlayers={self.layers}\n'

    def forward(self, input_: np.ndarray) -> np.ndarray:
        """Do forward propagation through all layers"""

        output = input_
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def add_layer(self, layer: LayerType, input_size: int = None, output_size: int = None, activation_func: Callable = None, activation_prime_func: Callable = None):
        """Add specified layer to the neural network"""

        match layer:
            case LayerType.FCL:
                if isinstance(input_size, int) and isinstance(output_size, int):
                    self.layers.append(FullyConnectedLayer(self.learning_rate, input_size, output_size))
                else:
                    raise ValueError('Incorrect argument for layer. Specify "input_size" and "output_size" as integers')

            case LayerType.ACTIVATION_LAYER:
                if isinstance(activation_func, Callable) and isinstance(activation_prime_func, Callable):
                    self.layers.append(ActivationLayer(activation_func, activation_prime_func))
                else:
                    raise ValueError('Incorrect argument for layer. Specify "activation_func" and "activation_func_prime" as callable functions')

    def train_step(self, input_data: np.ndarray, expected_output: np.ndarray):
        """Train the neural network"""

        predicted_output = self.forward(input_data)

        error = self.loss_func.func(expected_output, predicted_output)
        error_grad = self.loss_func.prime_func(expected_output, predicted_output)
        print(f'{error=}, {error_grad=}, {predicted_output=}')

        output_err_grad = error_grad
        for layer in reversed(self.layers):
            output_err_grad = layer.backward(output_err_grad)

    def save_model(self, filename: str):
        """Save model to pickle file"""

        if not os.path.splitext(filename)[1] == '.pkl':
            raise ValueError('Please save to a pickle file with the extension ".pkl"')

        model_data = {
            'learning_rate': self.learning_rate,
            'loss_func': self.loss_func,
            'layers': self.layers
        }

        with open(filename, 'wb') as file:
            pickle.dump(model_data, file)

    @classmethod
    def load_model(cls, filename: str):
        """Load model from pickle file"""

        if not os.path.isfile(filename) or not os.path.splitext(filename)[1] == '.pkl':
            raise ValueError('Please use a valid pickle file with the extension ".pkl"')

        with open(filename, 'rb') as file:
            model_data = pickle.load(file)

        learning_rate = model_data["learning_rate"]
        loss_func = model_data["loss_func"]
        layers = model_data["layers"]

        return Model(learning_rate=learning_rate, loss_func=loss_func, layers=layers)
