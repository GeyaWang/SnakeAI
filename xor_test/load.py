from neural_network import Model
import numpy as np


def get_input(model):
    while True:
        input_ = input('\nEnter an input:\n')
        output = model.predict(np.matrix(np.fromstring(input_, dtype=float, sep=',')))
        print(f'{input_} -> {output}')


def main():
    model = Model.load_model('training_data.pkl')
    get_input(model)


if __name__ == '__main__':
    main()
