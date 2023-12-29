from neural_network import Model, LayerType, MeanSquaredError, Tanh
import numpy as np


def train(model: Model):
    inputs = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    outputs = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    for i in range(1000):
        # batch training
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
