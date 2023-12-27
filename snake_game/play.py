import sys
import pygame
from game import Game, Direction
from threading import Thread
from collections import deque


WIDTH = 10
HEIGHT = 10
TILE_SIZE = 20
GAP_SIZE = 2
FPS = 60
SNAKE_MOVE_RATE = 7
N_STORED_INPUTS = 3
SNAKE_COLOUR = (255, 255, 255)
APPLE_COLOUR = (255, 0, 0)
BACKGROUND_COLOUR = (50, 50, 50)


class Draw:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen

    def tile(self, coord: tuple[int, int], colour: tuple[int, int, int]):
        x = coord[0] * (TILE_SIZE + GAP_SIZE) + GAP_SIZE
        y = coord[1] * (TILE_SIZE + GAP_SIZE) + GAP_SIZE
        pygame.draw.rect(self.screen, colour, (x, y, TILE_SIZE, TILE_SIZE))

    def background(self):
        self.screen.fill(BACKGROUND_COLOUR)


class Play:
    def __init__(self, screen_width: int, screen_height: int):
        screen = pygame.display.set_mode((screen_width, screen_height))

        self.clock = pygame.time.Clock()
        self.game = Game(WIDTH, HEIGHT)
        self.draw = Draw(screen)
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
                print('GAME OVER')
                return

            self.clock.tick(SNAKE_MOVE_RATE)

    def run(self):
        """Main game loop"""

        # game loop
        while True:
            # get inputs
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    match event.key:
                        case pygame.K_UP:
                            self.inputs.append(Direction.UP)
                        case pygame.K_DOWN:
                            self.inputs.append(Direction.DOWN)
                        case pygame.K_LEFT:
                            self.inputs.append(Direction.LEFT)
                        case pygame.K_RIGHT:
                            self.inputs.append(Direction.RIGHT)

            # draw
            self.draw.background()

            for x, y in self.game.snake_body:
                self.draw.tile((x, y), SNAKE_COLOUR)

            self.draw.tile(self.game.apple_pos, APPLE_COLOUR)

            pygame.display.flip()
            self.clock.tick(FPS)


def main():
    pygame.init()

    screen_width = WIDTH * (TILE_SIZE + GAP_SIZE) + GAP_SIZE
    screen_height = HEIGHT * (TILE_SIZE + GAP_SIZE) + GAP_SIZE

    Play(screen_width, screen_height).run()


if __name__ == '__main__':
    main()
