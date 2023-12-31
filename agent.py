from neural_network import Model, MeanSquaredError, LayerType, Tanh
from snake_game import Game, Direction
import random
import numpy as np
from collections import deque
from datetime import datetime, timedelta


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
    def __init__(
            self,
            initial_epsilon,
            epsilon_decay,
            min_epsilon,
            update_target_model_freq,
            max_memory_buffer,
            max_time_since_apple,
            n_episodes,
            gamma,
            batch_size,
            collision_detection_r,
            update_milestone,
            save_model_milestone,
            model_save_path,
            width,
            height,
            model,
            game
    ):
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.update_target_model_freq = update_target_model_freq
        self.max_memory_buffer = max_memory_buffer
        self.max_time_since_apple = max_time_since_apple
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.batch_size = batch_size
        self.collision_detection_r = collision_detection_r
        self.update_milestone = update_milestone
        self.save_model_milestone = save_model_milestone
        self.model_save_path = model_save_path
        self.width = width
        self.height = height
        self.main_model = model
        self.target_model = self.main_model.copy()
        self.game = game

        self.replay_memory = deque(maxlen=self.max_memory_buffer)
        self.running_score_list = deque(maxlen=self.update_milestone)

    def decay_epsilon(self):
        """Reduce epsilon to encourage exploitation"""

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

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
            reward -= 10  # punishment for death

        if is_eaten_apple:
            reward += 10  # reward eating apple

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
        elif (x, y) in self.game.snake_body or x >= self.width or x < 0 or y >= self.height or y < 0:
            return 0
        else:
            return 0.5

    def get_state(self) -> np.ndarray:
        """Returns simplified game state"""

        # list of bools if collision in square around snake head, match snake direction
        match self.game.direction:
            case Direction.UP:
                collision_list = [
                    self.get_tile_state(x, y)
                    for x in range(self.game.head_pos[0] - self.collision_detection_r, self.game.head_pos[0] + self.collision_detection_r + 1)
                    for y in range(self.game.head_pos[1] - self.collision_detection_r, self.game.head_pos[1] + self.collision_detection_r + 1)
                    if (x, y) != self.game.head_pos
                ]
            case Direction.RIGHT:
                collision_list = [
                    self.get_tile_state(x, y)
                    for y in range(self.game.head_pos[1] - self.collision_detection_r, self.game.head_pos[1] + self.collision_detection_r + 1)
                    for x in range(self.game.head_pos[0] + self.collision_detection_r, self.game.head_pos[0] - self.collision_detection_r - 1, -1)
                    if (x, y) != self.game.head_pos
                ]
            case Direction.DOWN:
                collision_list = [
                    self.get_tile_state(x, y)
                    for x in range(self.game.head_pos[0] + self.collision_detection_r, self.game.head_pos[0] - self.collision_detection_r - 1, -1)
                    for y in range(self.game.head_pos[1] + self.collision_detection_r, self.game.head_pos[1] - self.collision_detection_r - 1, -1)
                    if (x, y) != self.game.head_pos
                ]
            case _:
                collision_list = [
                    self.get_tile_state(x, y)
                    for y in range(self.game.head_pos[1] + self.collision_detection_r, self.game.head_pos[1] - self.collision_detection_r - 1, -1)
                    for x in range(self.game.head_pos[0] - self.collision_detection_r, self.game.head_pos[0] + self.collision_detection_r + 1)
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

        target = reward_arr + self.gamma * np.max(self.target_model.predict(next_state_arr), axis=1)
        target_q_values = self.main_model.predict(state_arr)
        action_index = np.array([self.index_to_action(direction_arr[i], action=action_arr[i], is_reversed=True) for i in range(batch_size)])
        target_q_values[0, action_index] = target

        self.main_model.train_step(state_arr, target_q_values)

    def save_model(self):
        print(f'Saved model to "{self.model_save_path}"')
        self.main_model.save_model(self.model_save_path)

    def run(self) -> tuple[list[int], list[float]]:
        """Training loop, return score list, reward list"""

        # record data to show progress of learning
        score_list = []
        avg_score_list = []

        # get initial timestamp
        initial_timestamp = datetime.now().timestamp()

        try:
            for episode in range(self.n_episodes):
                is_game_over = False
                score = 0
                time_since_apple = 0

                self.game.reset()
                state = self.get_state()

                while not is_game_over:
                    # reduce epsilon
                    self.decay_epsilon()

                    direction = self.game.direction

                    # use epsilon greedy policy to choose action
                    action = self.get_action(state)

                    # play game step
                    is_game_over, is_eaten_apple = self.game.play_step(action)

                    if is_eaten_apple:
                        score += 1
                        time_since_apple = 0
                    else:
                        time_since_apple += 1

                    # if too long since eaten apple, stop
                    if time_since_apple >= self.max_time_since_apple:
                        break

                    # calculate reward
                    reward = self.get_reward(is_game_over, is_eaten_apple)

                    # continue to next step
                    next_state = self.get_state()

                    # train step
                    batch = (state, direction, action, reward, next_state)
                    self.replay_memory.append(batch)
                    self.update_model([batch], 1)

                    state = next_state

                """Game Over"""

                # select batch from replay buffer and train
                n_replays = len(self.replay_memory)
                if n_replays >= self.batch_size:
                    batch = random.sample(self.replay_memory, self.batch_size)
                    self.update_model(batch, self.batch_size)

                # record data
                self.running_score_list.append(score)
                score_list.append(score)

                # update target model if multiple of update frequency
                if episode % self.update_target_model_freq == 0:
                    self.target_model = self.main_model.copy()

                # print update every milestone
                if episode % self.update_milestone == 0 and episode != 0:
                    average_score = sum(self.running_score_list) / len(self.running_score_list)
                    timestamp = datetime.now().timestamp()

                    print(f'[Episode {episode}] avg score: {average_score}, time elapsed: {timedelta(seconds=timestamp - initial_timestamp)}')

                    avg_score_list.append(average_score)

                # save model every milestone
                if episode % self.save_model_milestone == 0 and episode != self.n_episodes - 1 and episode != 0:
                    self.save_model()

        except KeyboardInterrupt:
            print('Forcefully shut down by user')

        # training finished
        self.save_model()

        return score_list, avg_score_list
