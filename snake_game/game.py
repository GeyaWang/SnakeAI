from enum import Enum
from collections import deque
import random


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class Game:
    def __init__(self, width: int, height: int, default_head_pos: tuple[int, int] = (0, 0), default_direction: Direction = Direction.RIGHT, default_len: int = 1):
        self.width = width
        self.height = height

        self.head_pos = default_head_pos
        self.direction = default_direction
        self.length = default_len

        self.snake_body = deque([self.head_pos], maxlen=default_len)
        self.apple_pos = self.spawn_apple()

    def spawn_apple(self) -> tuple[int, int]:
        """Keep randomly generating position until not inside snake"""

        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)

            if (x, y) not in self.snake_body:
                return x, y

    def play_step(self, action: Direction = None) -> tuple[bool, bool]:
        """Play game step, returns is_game_over, is_eaten_apple"""

        is_game_over = False
        is_eaten_apple = False

        if action is None:
            action = self.direction
        else:
            self.direction = action

        # move
        new_head_pos = (self.head_pos[0] + action.value[0], self.head_pos[1] + action.value[1])

        # check if dead
        # # done before new head added
        if new_head_pos in self.snake_body or new_head_pos[0] >= self.width or new_head_pos[0] < 0 or new_head_pos[1] >= self.height or new_head_pos[1] < 0:
            is_game_over = True

        # finish moving
        self.snake_body.append(new_head_pos)
        self.head_pos = new_head_pos

        # check if eaten apple
        if new_head_pos == self.apple_pos:
            is_eaten_apple = True

            # spawn new apple
            self.apple_pos = self.spawn_apple()

            # increase snake length
            self.length += 1
            self.snake_body = deque(self.snake_body, maxlen=self.length)

        return is_game_over, is_eaten_apple
