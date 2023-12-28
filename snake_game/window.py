import pygame
from .game import Game
import sys


class Draw:
    def __init__(
            self,
            game: Game,
            tile_size: int,
            gap_size: int,
            background_colour: tuple[int, int, int],
            snake_colour: tuple[int, int, int],
            apple_colour: tuple[int, int, int]
    ):
        self.game = game
        self.tile_size = tile_size
        self.gap_size = gap_size
        self.background_colour = background_colour
        self.snake_colour = snake_colour
        self.apple_colour = apple_colour

        self.screen = pygame.display.get_surface()

    def tile(self, coord: tuple[int, int], colour: tuple[int, int, int]):
        x = coord[0] * (self.tile_size + self.gap_size) + self.gap_size
        y = coord[1] * (self.tile_size + self.gap_size) + self.gap_size
        pygame.draw.rect(self.screen, colour, (x, y, self.tile_size, self.tile_size))

    def background(self):
        self.screen.fill(self.background_colour)

    def draw(self):
        # background
        self.background()

        # snake
        for coord in self.game.snake_body:
            self.tile(coord, self.snake_colour)

        # apple
        self.tile(self.game.apple_pos, self.apple_colour)

        pygame.display.flip()


class Window:
    def __init__(
            self,
            fps: int,
            game: Game,
            width: int,
            height: int,
            tile_size: int,
            gap_size: int,
            background_colour: tuple[int, int, int] = (50, 50, 50),
            snake_colour: tuple[int, int, int] = (255, 255, 255),
            apple_colour: tuple[int, int, int] = (255, 0, 0)
    ):
        screen_width = width * (tile_size + gap_size) + gap_size
        screen_height = height * (tile_size + gap_size) + gap_size
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.fps = fps
        self.clock = pygame.time.Clock()

        self.game = game,
        self.draw = Draw(game, tile_size, gap_size, background_colour, snake_colour, apple_colour)

    def update(self) -> list[pygame.event.Event]:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                sys.exit()

        self.draw.draw()
        self.clock.tick(self.fps)

        return events
