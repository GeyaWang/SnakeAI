from neural_network import Model, LayerType, MeanSquaredError, Tanh
import numpy as np
import random


def train(model: Model):
    training_data = [
        (np.array([[0, 0]]), np.array([[0,]])),
        (np.array([[1, 0]]), np.array([[1,]])),
        (np.array([[0, 1]]), np.array([[1,]])),
        (np.array([[1, 1]]), np.array([[0,]]))
    ]

    inputs = np.array([
        np.array([[0, 0]]),
        np.array([[1, 0]]),
        np.array([[0, 1]]),
        np.array([[1, 1]])
    ])

    outputs = np.array([
        np.array([[0, ]]),
        np.array([[1, ]]),
        np.array([[1, ]]),
        np.array([[0, ]])
    ])

    for i in range(1000):
        # input_data, expected_output = random.choice(training_data)
        # print(f'\n{input_data=}, {expected_output=}')
        # model.train_step(input_data, expected_output)
        model.train_step(inputs, outputs)

    model.save_model('training_data.pkl')


def main():
    model = Model(learning_rate=0.1, loss_func=MeanSquaredError)

    model.add_layer(LayerType.FCL, input_size=2, output_size=15)
    model.add_layer(LayerType.ACTIVATION_LAYER, activation_func=Tanh)
    model.add_layer(LayerType.FCL, input_size=15, output_size=15)
    model.add_layer(LayerType.ACTIVATION_LAYER, activation_func=Tanh)
    model.add_layer(LayerType.FCL, input_size=15, output_size=1)
    print(model)

    train(model)


if __name__ == '__main__':
    main()
