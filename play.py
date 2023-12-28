from snake_game import Game, Window, Direction
import time
from threading import Thread
from collections import deque
import pygame


WIDTH = 21
HEIGHT = 21
TILE_SIZE = 20
GAP_SIZE = 2
FPS = 60
SNAKE_MOVE_RATE = 7
N_STORED_INPUTS = 3


class Play:
    def __init__(self):
        self.game = Game(WIDTH, HEIGHT, default_head_pos=(5, 10))
        self.window = Window(FPS, self.game, WIDTH, HEIGHT, TILE_SIZE, GAP_SIZE)
        self.inputs = deque(maxlen=N_STORED_INPUTS)

        self.score = 0

        input_thread = Thread(target=self.update_game, daemon=True)
        input_thread.start()

    def update_game(self):
        """Thread play game step"""

        while True:
            if self.inputs:
                input_ = self.inputs[0]
                self.inputs.popleft()
            else:
                input_ = None

            is_game_over, is_eaten_apple = self.game.play_step(input_)

            if is_eaten_apple:
                self.score += 1
                print(f'NEW SCORE: {self.score}')

            if is_game_over:
                print('GAME OVER\n')
                self.game.reset()
                self.score = 0

            time.sleep(1 / SNAKE_MOVE_RATE)

    def run(self):
        """Main game loop"""

        # game loop
        while True:
            # update window
            events = self.window.update()

            # get inputs
            for event in events:
                if event.type == pygame.KEYDOWN:
                    match event.key:
                        case pygame.K_UP:
                            self.inputs.append(Direction.UP)
                        case pygame.K_DOWN:
                            self.inputs.append(Direction.DOWN)
                        case pygame.K_LEFT:
                            self.inputs.append(Direction.LEFT)
                        case pygame.K_RIGHT:
                            self.inputs.append(Direction.RIGHT)


def main():
    Play().run()


if __name__ == '__main__':
    main()
