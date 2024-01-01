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
SNAKE_MOVE_RATE = 120

TEMP_VIDEO_DIR = '__temp__'
IS_CREATE_VIDEO = True
OUTPUT_VIDEO_PATH = 'snake.mp4'


class PlayAI:
    def __init__(self):
        self.game = Game(WIDTH, HEIGHT, default_head_pos=(5, 10))
        self.window = Window(FPS, self.game, TILE_SIZE, GAP_SIZE)
        self.agent = Agent(model=Model.load_model('snake_model.pkl'), game=self.game)

        self.is_running = True

        self.frames = 0
        self.score = 0

        game_thread = Thread(target=self.update_game, daemon=True)
        game_thread.start()

        if IS_CREATE_VIDEO:
            create_video_dir()

    def game_over(self):
        print('GAME OVER\n')

        if IS_CREATE_VIDEO:
            image_arr = []
            for frame in range(self.frames):
                # get image
                image = cv2.imread(f'{TEMP_VIDEO_DIR}/{frame}.png')
                image_arr.append(image)

                # delete image
                os.remove(f'{TEMP_VIDEO_DIR}/{frame}.png')

            # delete directory
            os.rmdir(TEMP_VIDEO_DIR)

            # create video
            size = (self.window.screen_width, self.window.screen_height)
            out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FPS, size)
            for image in image_arr:
                out.write(image)
            out.release()

        self.window.close()
        sys.exit()

    def update_game(self):
        """Thread to update game at independant rate"""

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

            if IS_CREATE_VIDEO:
                self.window.save_image(f'{TEMP_VIDEO_DIR}/{self.frames}.png')

            self.frames += 1

        self.game_over()


def create_video_dir():
    os.makedirs(TEMP_VIDEO_DIR)


def main():
    PlayAI().run()


if __name__ == '__main__':
    main()
