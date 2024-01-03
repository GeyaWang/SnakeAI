from matplotlib import pyplot as plt
from agent import Agent
from snake_game import Game
from neural_network import Model, MeanSquaredError, Tanh, LayerType
from config import *
import sys


WIDTH = 21
HEIGHT = 21
IS_USE_PREV_MODEL = True


def plot_graph(score_list: list[int], avg_score_list: list[float]):
    """plot graph of progress"""

    plt.plot([i for i in range(len(score_list))], score_list)
    plt.plot([i * UPDATE_MILESTONE for i in range(len(avg_score_list))], avg_score_list)

    plt.ylabel(f'Score')
    plt.xlabel('Episode')

    plt.savefig('snake_progress.png')


def main():
    game = Game(WIDTH, HEIGHT, default_head_pos=(5, 10))

    if IS_USE_PREV_MODEL:
        model = Model.load_model(MODEL_SAVE_PATH)
    else:
        model = Model(learning_rate=LEARNING_RATE, loss_func=MeanSquaredError)
        model.add_layer(LayerType.FCL, input_size=7 + ((COLLISION_DETECTION_RADIUS * 2) + 1) ** 2, output_size=256)
        model.add_layer(LayerType.ACTIVATION_LAYER, activation_func=Tanh)
        model.add_layer(LayerType.FCL, input_size=256, output_size=256)
        model.add_layer(LayerType.ACTIVATION_LAYER, activation_func=Tanh)
        model.add_layer(LayerType.FCL, input_size=256, output_size=3)

    agent = Agent(model, game)
    score_list, avg_score_list = agent.run()

    plot_graph(score_list, avg_score_list)

    sys.exit()


if __name__ == '__main__':
    main()
