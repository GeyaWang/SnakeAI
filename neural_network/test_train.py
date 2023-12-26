from model import Model
from layer import LayerType
from activation_func import tanh_func, tanh_prime_func
from loss_func import MeanSquaredError
import numpy as np
import random


def train(model: Model):
    training_data = [
        (np.array([[0, 0]]), np.array([[0,]])),
        (np.array([[1, 0]]), np.array([[1,]])),
        (np.array([[0, 1]]), np.array([[1,]])),
        (np.array([[1, 1]]), np.array([[0,]]))
    ]

    for i in range(10000):
        input_data, expected_output = random.choice(training_data)
        print(f'\n{input_data=}, {expected_output=}')
        model.train_step(input_data, expected_output)

    model.save_model('training_data.pkl')


def main():
    model = Model(learning_rate=0.1, loss_func=MeanSquaredError)

    model.add_layer(LayerType.FCL, input_size=2, output_size=15)
    model.add_layer(LayerType.ACTIVATION_LAYER, activation_func=tanh_func, activation_prime_func=tanh_prime_func)
    model.add_layer(LayerType.FCL, input_size=15, output_size=15)
    model.add_layer(LayerType.ACTIVATION_LAYER, activation_func=tanh_func, activation_prime_func=tanh_prime_func)
    model.add_layer(LayerType.FCL, input_size=15, output_size=1)
    print(model)

    train(model)


if __name__ == '__main__':
    main()
