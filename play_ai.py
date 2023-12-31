from snake_game import Game, Window
from neural_network import Model
import time
from threading import Thread
from agent import Agent
import sys
import cv2
import os

WIDTH = 21
HEIGHT = 21
TILE_SIZE = 20
GAP_SIZE = 2
FPS = 60
SNAKE_MOVE_RATE = 60


class PlayAI:
    def __init__(self):
        self.game = Game(WIDTH, HEIGHT, default_head_pos=(5, 10))
        self.window = Window(FPS, self.game, TILE_SIZE, GAP_SIZE)
        self.agent = Agent(model=Model.load_model('snake_model.pkl'), game=self.game)

        self.is_running = True
        self.score = 0

        game_thread = Thread(target=self.update_game, daemon=True)
        game_thread.start()

    def update_game(self):
        """Thread to update game at independent rate"""

        while self.is_running:
            state = self.agent.get_state()
            action = self.agent.get_action(state, epsilon=0)

            is_game_over, is_eaten_apple = self.game.play_step(action)

            if is_eaten_apple:
                self.score += 1
                print(f'NEW SCORE: {self.score}')

            if is_game_over:
                self.is_running = False

            time.sleep(1 / SNAKE_MOVE_RATE)

    def run(self):
        """Game loop"""

        while self.is_running:
            self.window.update()

        # game over
        print('GAME OVER\n')
        self.window.close()
        sys.exit()


def main():
    PlayAI().run()


if __name__ == '__main__':
    main()
