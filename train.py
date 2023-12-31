from matplotlib import pyplot as plt
from agent import Agent
from threading import Thread
from snake_game import Window, Game
from neural_network import Model, MeanSquaredError, Tanh, LayerType
import sys


LEARNING_RATE = 0.003
INITIAL_EPSILON = 1
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01
UPDATE_TARGET_MODEL_FREQUENCY = 300
MAX_MEMORY_BUFFER = 10_000
MAX_TIME_SINCE_APPLE = 2_500
N_EPISODES = 101
GAMMA = 0.99
BATCH_SIZE = 256
COLLISION_DETECTION_RADIUS = 4
UPDATE_MILESTONE = 100
SAVE_MODEL_MILESTONE = 1000
MODEL_SAVE_PATH = 'snake_model.pkl'

WIDTH = 21
HEIGHT = 21
TILE_SIZE = 20
GAP_SIZE = 2
FPS = 120
IS_SHOW_WINDOW = True


def plot_graph(score_list: list[int], avg_score_list: list[float]):
    """plot graph of progress"""

    plt.plot([i for i in range(len(score_list))], score_list)
    plt.plot([i * UPDATE_MILESTONE for i in range(len(avg_score_list))], avg_score_list)

    plt.ylabel(f'Score')
    plt.xlabel('Episode')

    plt.savefig('snake_progress.png')


def game_thread(game: Game):
    window = Window(FPS, game, TILE_SIZE, GAP_SIZE)
    while True:
        window.update()


def main():
    game = Game(WIDTH, HEIGHT, default_head_pos=(5, 10))

    model = Model(learning_rate=LEARNING_RATE, loss_func=MeanSquaredError)
    model.add_layer(LayerType.FCL, input_size=7 + ((COLLISION_DETECTION_RADIUS * 2) + 1) ** 2, output_size=256)
    model.add_layer(LayerType.ACTIVATION_LAYER, activation_func=Tanh)
    model.add_layer(LayerType.FCL, input_size=256, output_size=256)
    model.add_layer(LayerType.ACTIVATION_LAYER, activation_func=Tanh)
    model.add_layer(LayerType.FCL, input_size=256, output_size=3)

    if IS_SHOW_WINDOW:
        t = Thread(target=game_thread, daemon=True, args=(game,))
        t.start()

    agent = Agent(
        INITIAL_EPSILON,
        EPSILON_DECAY,
        MIN_EPSILON,
        UPDATE_TARGET_MODEL_FREQUENCY,
        MAX_MEMORY_BUFFER,
        MAX_TIME_SINCE_APPLE,
        N_EPISODES,
        GAMMA,
        BATCH_SIZE,
        COLLISION_DETECTION_RADIUS,
        UPDATE_MILESTONE,
        SAVE_MODEL_MILESTONE,
        MODEL_SAVE_PATH,
        WIDTH,
        HEIGHT,
        model,
        game
    )
    score_list, avg_score_list = agent.run()

    plot_graph(score_list, avg_score_list)

    sys.exit()


if __name__ == '__main__':
    main()
