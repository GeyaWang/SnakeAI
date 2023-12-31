from neural_network import Model, MeanSquaredError, LayerType, Tanh
from snake_game import Game, Direction, Window
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime, timedelta

LEARNING_RATE = 0.003
INITIAL_EPSILON = 1
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01
UPDATE_TARGET_MODEL_FREQUENCY = 300
MAX_MEMORY_BUFFER = 10_000
N_EPISODES = 100_000
GAMMA = 0.99
BATCH_SIZE = 512
COLLISION_DETECTION_RADIUS = 4
UPDATE_MILESTONE = 100
SAVE_MODEL_MILESTONE = 1000
MODEL_SAVE_PATH = 'snake_model.pkl'

WIDTH = 21
HEIGHT = 21
TILE_SIZE = 20
GAP_SIZE = 2
FPS = 30
IS_SHOW_WINDOW = False


def rotate_direction(direction: Direction, is_clockwise: bool = None):
    directions = list(Direction)

    if is_clockwise is None:
        raise ValueError('Please specify a direction to rotate')

    if is_clockwise:
        index = (directions.index(direction) + 1) % len(directions)
    else:  # not is_clockwise
        index = (directions.index(direction) - 1) % len(directions)

    return directions[index]


class Agent:
    def __init__(self, model: Model = None, game: Game = None):
        if model is None:
            self.main_model = Model(learning_rate=LEARNING_RATE, loss_func=MeanSquaredError)
            self.main_model.add_layer(LayerType.FCL, input_size=7 + ((COLLISION_DETECTION_RADIUS * 2) + 1) ** 2, output_size=256)
            self.main_model.add_layer(LayerType.ACTIVATION_LAYER, activation_func=Tanh)
            self.main_model.add_layer(LayerType.FCL, input_size=256, output_size=256)
            self.main_model.add_layer(LayerType.ACTIVATION_LAYER, activation_func=Tanh)
            self.main_model.add_layer(LayerType.FCL, input_size=256, output_size=256)
            self.main_model.add_layer(LayerType.ACTIVATION_LAYER, activation_func=Tanh)
            self.main_model.add_layer(LayerType.FCL, input_size=256, output_size=3)
            self.target_model = self.main_model.copy()
        else:
            self.main_model = model

        if game is None:
            self.game = Game(WIDTH, HEIGHT, default_head_pos=(5, 10))
        else:
            self.game = game

        if IS_SHOW_WINDOW:
            self.window = Window(FPS, self.game, WIDTH, HEIGHT, TILE_SIZE, GAP_SIZE)

        self.epsilon = INITIAL_EPSILON
        self.replay_memory = deque(maxlen=MAX_MEMORY_BUFFER)
        self.running_score_list = deque(maxlen=UPDATE_MILESTONE)

    def decay_epsilon(self):
        """Reduce epsilon to encourage exploitation"""

        self.epsilon = max(self.epsilon * EPSILON_DECAY, MIN_EPSILON)

    @staticmethod
    def index_to_action(direction: Direction, index: int = None, action: Direction = None, is_reversed: bool = False):
        """Converts action index to direction and vise versa"""

        left = rotate_direction(direction, is_clockwise=False)
        right = rotate_direction(direction, is_clockwise=True)

        action_list = [left, direction, right]

        if not is_reversed:
            return action_list[index]
        else:  # if is_reversed
            return action_list.index(action)

    @staticmethod
    def get_reward(is_game_over: bool, is_eaten_apple: bool) -> int:
        """Assign reward values for survival, death and eating apple"""

        reward = 0

        if is_game_over:
            reward -= 1  # punishment for death

        if is_eaten_apple:
            reward += 1  # reward eating apple

        return reward

    def get_action(self, state: np.array, epsilon: float = None) -> Direction:
        """Epsilon greedy policy"""

        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:  # exploration
            # randomly choose between going left, right of forwards
            action_index = random.randint(0, 2)
        else:  # exploitation
            # convert list of bools to matrix of 1s and 0s and input into model
            predicted_output = self.main_model.predict(state)

            # get action from the largest value in output matrix
            action_index = np.argmax(predicted_output)

        return self.index_to_action(self.game.direction, action_index)

    def get_tile_state(self, x: int, y: int) -> float:
        """Returns value based on tile state"""

        if self.game.apple_pos == (x, y):
            return 1
        elif (x, y) in self.game.snake_body or x >= WIDTH or x < 0 or y >= HEIGHT or y < 0:
            return 0
        else:  # empty space
            return 0.5

    def get_state(self) -> np.ndarray:
        """Returns simplified game state"""

        # list of bools if collision in square around snake head, match snake direction
        match self.game.direction:
            case Direction.UP:
                collision_list = [
                    self.get_tile_state(x, y)
                    for x in range(self.game.head_pos[0] - COLLISION_DETECTION_RADIUS, self.game.head_pos[0] + COLLISION_DETECTION_RADIUS + 1)
                    for y in range(self.game.head_pos[1] - COLLISION_DETECTION_RADIUS, self.game.head_pos[1] + COLLISION_DETECTION_RADIUS + 1)
                    if (x, y) != self.game.head_pos
                ]
            case Direction.RIGHT:
                collision_list = [
                    self.get_tile_state(x, y)
                    for y in range(self.game.head_pos[1] - COLLISION_DETECTION_RADIUS, self.game.head_pos[1] + COLLISION_DETECTION_RADIUS + 1)
                    for x in range(self.game.head_pos[0] + COLLISION_DETECTION_RADIUS, self.game.head_pos[0] - COLLISION_DETECTION_RADIUS - 1, -1)
                    if (x, y) != self.game.head_pos
                ]
            case Direction.DOWN:
                collision_list = [
                    self.get_tile_state(x, y)
                    for x in range(self.game.head_pos[0] + COLLISION_DETECTION_RADIUS, self.game.head_pos[0] - COLLISION_DETECTION_RADIUS - 1, -1)
                    for y in range(self.game.head_pos[1] + COLLISION_DETECTION_RADIUS, self.game.head_pos[1] - COLLISION_DETECTION_RADIUS - 1, -1)
                    if (x, y) != self.game.head_pos
                ]
            case _:
                collision_list = [
                    self.get_tile_state(x, y)
                    for y in range(self.game.head_pos[1] + COLLISION_DETECTION_RADIUS, self.game.head_pos[1] - COLLISION_DETECTION_RADIUS - 1, -1)
                    for x in range(self.game.head_pos[0] - COLLISION_DETECTION_RADIUS, self.game.head_pos[0] + COLLISION_DETECTION_RADIUS + 1)
                    if (x, y) != self.game.head_pos
                ]

        is_direction_up = self.game.direction == Direction.UP
        is_direction_right = self.game.direction == Direction.RIGHT
        is_direction_down = self.game.direction == Direction.DOWN
        is_direction_left = self.game.direction == Direction.LEFT
        is_apple_up = self.game.apple_pos[1] < self.game.head_pos[1]  # (0, 0) is top left
        is_apple_right = self.game.apple_pos[0] > self.game.head_pos[0]
        is_apple_down = self.game.apple_pos[1] > self.game.head_pos[1]
        is_apple_left = self.game.apple_pos[0] < self.game.head_pos[0]

        state = np.array([
            *collision_list,
            is_direction_up,
            is_direction_right,
            is_direction_down,
            is_direction_left,
            is_apple_up,
            is_apple_right,
            is_apple_down,
            is_apple_left
        ], dtype=float)

        return state

    def update_model(self, batch: list[tuple], batch_size: int):
        """Train model, comparing predicted Q value to target Q value from the Bellman equation"""

        state_list, direction_list, action_list, reward_list, next_state_list = zip(*batch)

        # convert to arrays
        state_arr, direction_arr, action_arr, reward_arr, next_state_arr = np.array(state_list), np.array(direction_list), np.array(action_list), np.array(reward_list), np.array(next_state_list)

        target = reward_arr + GAMMA * np.max(self.target_model.predict(next_state_arr), axis=1)
        target_q_values = self.main_model.predict(state_arr)
        action_index = np.array([self.index_to_action(direction_arr[i], action=action_arr[i], is_reversed=True) for i in range(batch_size)])
        target_q_values[0, action_index] = target

        self.main_model.train_step(state_arr, target_q_values)

    def save_model(self):
        print(f'Saved model to "{MODEL_SAVE_PATH}"')
        self.main_model.save_model(MODEL_SAVE_PATH)

    def run(self) -> tuple[list[int], list[float]]:
        """Training loop, return score list, reward list"""

        # record data to show progress of learning
        score_list = []
        avg_score_list = []

        # get initial timestamp
        initial_timestamp = datetime.now().timestamp()

        try:
            for episode in range(N_EPISODES):
                is_game_over = False
                score = 0

                self.game.reset()
                state = self.get_state()

                while not is_game_over:
                    self.decay_epsilon()

                    direction = self.game.direction
                    action = self.get_action(state)
                    is_game_over, is_eaten_apple = self.game.play_step(action)

                    if is_eaten_apple:
                        score += 1

                    reward = self.get_reward(is_game_over, is_eaten_apple)

                    next_state = self.get_state()

                    batch = (state, direction, action, reward, next_state)
                    self.replay_memory.append(batch)
                    self.update_model([batch], 1)

                    state = next_state

                    if IS_SHOW_WINDOW:
                        self.window.update()

                # game over

                # select batch from replay buffer and train
                n_replays = len(self.replay_memory)
                if n_replays >= BATCH_SIZE:
                    batch = random.sample(self.replay_memory, BATCH_SIZE)
                    self.update_model(batch, BATCH_SIZE)

                self.running_score_list.append(score)
                score_list.append(score)

                # update target model if multiple of update frequency
                if episode % UPDATE_TARGET_MODEL_FREQUENCY == 0:
                    self.target_model = self.main_model.copy()

                # print update every milestone
                if episode % UPDATE_MILESTONE == 0 and episode != 0:
                    average_score = sum(self.running_score_list) / len(self.running_score_list)
                    timestamp = datetime.now().timestamp()

                    print(f'[Episode {episode}] avg score: {average_score}, time elapsed: {timedelta(seconds=timestamp - initial_timestamp)}')

                    avg_score_list.append(average_score)

                # save model every milestone
                if episode % SAVE_MODEL_MILESTONE == 0 and episode != N_EPISODES - 1 and episode != 0:
                    self.save_model()

        except KeyboardInterrupt:
            print('Forcefully shut down by user')

        # training finished
        self.save_model()

        return score_list, avg_score_list


def plot_graph(score_list: list[int], avg_score_list: list[float]):
    """plot graph of progress"""

    plt.plot([i for i in range(len(score_list))], score_list)
    plt.plot([i * UPDATE_MILESTONE for i in range(len(avg_score_list))], avg_score_list)

    plt.ylabel(f'Score')
    plt.xlabel('Episode')

    plt.savefig('snake_progress.png')


def main():
    agent = Agent()
    score_list, avg_score_list = agent.run()

    plot_graph(score_list, avg_score_list)


if __name__ == '__main__':
    main()
