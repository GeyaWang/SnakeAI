from neural_network import Model, MeanSquaredError, LayerType, Tanh
from snake_game import Game, Direction, Window
import random
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.005
INITIAL_EPSILON = 1
EPSILON_DECAY = 0.996
MIN_EPSILON = 0.1
UPDATE_TARGET_MODEL_FREQUENCY = 200
N_EPISODES = 2000
BATCH_SIZE = 32
GAMMA = 0.99

WIDTH = 21
HEIGHT = 21
TILE_SIZE = 20
GAP_SIZE = 2
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
            self.main_model.add_layer(LayerType.FCL, input_size=11, output_size=32)
            self.main_model.add_layer(LayerType.ACTIVATION_LAYER, activation_func=Tanh)
            self.main_model.add_layer(LayerType.FCL, input_size=32, output_size=24)
            self.main_model.add_layer(LayerType.ACTIVATION_LAYER, activation_func=Tanh)
            self.main_model.add_layer(LayerType.FCL, input_size=24, output_size=3)
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

    def decay_epsilon(self):
        """Reduce epsilon to encourage exploitation"""

        self.epsilon = max(self.epsilon * EPSILON_DECAY, MIN_EPSILON)

    def get_state(self) -> np.ndarray:
        """Returns simplified game state"""

        is_collision_left = self.game.simulate_step(rotate_direction(self.game.direction, is_clockwise=False))[0]
        is_collision_straight = self.game.simulate_step(self.game.direction)[0]
        is_collision_right = self.game.simulate_step(rotate_direction(self.game.direction, is_clockwise=True))[0]
        is_direction_up = self.game.direction == Direction.UP
        is_direction_right = self.game.direction == Direction.RIGHT
        is_direction_down = self.game.direction == Direction.DOWN
        is_direction_left = self.game.direction == Direction.LEFT
        is_apple_up = self.game.apple_pos[1] < self.game.head_pos[1]  # (0, 0) is top left
        is_apple_right = self.game.apple_pos[0] > self.game.head_pos[0]
        is_apple_down = self.game.apple_pos[1] > self.game.head_pos[1]
        is_apple_left = self.game.apple_pos[0] < self.game.head_pos[0]

        state = np.array([
            is_collision_left,
            is_collision_straight,
            is_collision_right,
            is_direction_up,
            is_direction_right,
            is_direction_down,
            is_direction_left,
            is_apple_up,
            is_apple_right,
            is_apple_down,
            is_apple_left
        ], dtype=int)

        return state

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

    @staticmethod
    def get_reward(is_game_over: bool, is_eaten_apple: bool, time_since_eaten_apple: int) -> int:
        """Assign reward values for survival, death and eating apple"""

        reward = 0

        if is_game_over:
            reward -= 20  # punishment for death
        else:
            if time_since_eaten_apple > 40:
                reward -= 1  # punish for just surviving, not eating apple
            else:
                reward += 1  # reward survival

        if is_eaten_apple:
            reward += 20  # reward eating apple

        return reward

    def update_model(self, state: np.matrix, direction: Direction, action: Direction, reward: int, next_state: np.matrix):
        """Train model, comparing predicted Q value to target Q value from the Bellman equation"""

        target = reward + GAMMA * np.max(self.target_model.predict(next_state))
        target_q_values = self.main_model.predict(state)
        action_index = self.index_to_action(direction, action=action, is_reversed=True)
        target_q_values[0, action_index] = target

        self.main_model.train_step(np.matrix(state), target_q_values)

    def run(self):
        """Training loop"""

        # record data to show progress of learning
        score_list = []
        reward_list = []

        for episode in range(N_EPISODES):
            is_game_over = False
            time_since_eaten_apple = 0
            score = 0
            total_reward = 0

            self.game.reset()
            state = self.get_state()

            while not is_game_over:
                self.decay_epsilon()

                direction = self.game.direction
                action = self.get_action(state)
                is_game_over, is_eaten_apple = self.game.play_step(action)

                if is_eaten_apple:
                    score += 1
                    time_since_eaten_apple = 0
                else:
                    time_since_eaten_apple += 1

                reward = self.get_reward(is_game_over, is_eaten_apple, time_since_eaten_apple)
                total_reward += reward

                next_state = self.get_state()

                self.update_model(np.matrix(state), direction, action, reward, np.matrix(next_state))

                state = next_state

                if IS_SHOW_WINDOW:
                    self.window.update()

            # game over
            print(f'{episode=}: {score=}, {total_reward=}')
            score_list.append(score)
            reward_list.append(total_reward)

            # update target model if multiple of update frequency
            if episode % UPDATE_TARGET_MODEL_FREQUENCY == 0:
                self.target_model = self.main_model.copy()

        # training finished
        self.main_model.save_model('snake_model.pkl')

        # plot graph of progress
        plt.subplot(2, 1, 1)
        plt.plot([i for i in range(N_EPISODES)], score_list)
        plt.ylabel('Score')

        plt.subplot(2, 1, 2)
        plt.plot([i for i in range(N_EPISODES)], reward_list)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        plt.savefig('snake_progress.png')


def main():
    agent = Agent()
    agent.run()


if __name__ == '__main__':
    main()
