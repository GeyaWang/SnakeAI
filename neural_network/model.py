import numpy as np
from .layer import LayerType, FullyConnectedLayer, ActivationLayer
from .loss_func import LossFunc
from .activation_func import ActivationFunction
from typing import Type
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

    def add_layer(self, layer: LayerType, input_size: int = None, output_size: int = None, activation_func: Type[ActivationFunction] = None):
        """Add specified layer to the neural network"""

        match layer:
            case LayerType.FCL:
                if isinstance(input_size, int) and isinstance(output_size, int):
                    self.layers.append(FullyConnectedLayer(self.learning_rate, input_size, output_size))
                else:
                    raise ValueError('Incorrect argument for "layer". Specify "input_size" and "output_size" as integers')

            case LayerType.ACTIVATION_LAYER:
                if isinstance(activation_func, type) and issubclass(activation_func, ActivationFunction):
                    self.layers.append(ActivationLayer(activation_func))
                else:
                    raise ValueError('Incorrect argument for "activation_func"')

    def predict(self, input_: np.ndarray) -> np.ndarray:
        """Do forward propagation through all layers"""

        output = input_
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def train_step(self, input_data: np.ndarray, expected_output: np.ndarray):
        """Train the neural network"""

        predicted_output = self.predict(input_data)

        error = self.loss_func.func(expected_output, predicted_output)
        error_grad = self.loss_func.prime_func(expected_output, predicted_output)
        print(f'{error=}, {error_grad=}, {predicted_output=}')

        # backward propagation
        output_err_grad = error_grad
        for layer in reversed(self.layers):
            output_err_grad = layer.backward(output_err_grad)

    def train_batch(self, input_data_arr: np.ndarray[np.ndarray], expected_output_arr: np.ndarray[np.ndarray]):
        """Train in a batch, accumulating the error gradients before backward propagation"""

        if len(input_data_arr) != len(expected_output_arr):
            raise ValueError('Lengths of "input_data_arr" and "expected_output_arr" do not match')

        total_error_grad = None

        for i in range(len(input_data_arr)):
            input_data = input_data_arr[i]
            expected_output = expected_output_arr[i]

            predicted_output = self.predict(input_data)
            error_grad = self.loss_func.prime_func(expected_output, predicted_output)

            # accumulate error gradients
            if total_error_grad is None:
                total_error_grad = error_grad
            else:
                total_error_grad = np.add(total_error_grad, error_grad)

        # backward propagation
        output_err_grad = total_error_grad
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
