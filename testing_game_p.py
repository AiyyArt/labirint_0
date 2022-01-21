"""
**************************************

Created in 21.01.2022
by Aiyyskhan Alekseev

https://github.com/AiyyArt
timirkhan@gmail.com

**************************************
"""

__author__ = "Aiyyskhan Alekseev"
__version__ = "2.2.0"

import math
import numpy as np
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import pygame

from settings import *
import nn_1pl as nn
from player_for_testing_game import Player
from map_file_lev0_2 import get_map
from drawing import Drawing
from ray_casting import RayCast

PDF_LOAD_PATH = "D:/Ganglion/Ganglion_v.1_torch/data/png_images/gen_18_1.png" 

class Game:
    def __init__(self):
        self.color_r = np.random.randint(50, 220) # np.linspace(10, 200).astype(int)
        self.color_g = np.random.randint(50, 220) # np.linspace(10, 200).astype(int)
        self.color_b = np.random.randint(50, 220) # np.linspace(200, 10).astype(int)

        pygame.init()
        self.sc = pygame.display.set_mode((WIDTH, HEIGHT))
        self.sc_map = pygame.Surface((WIDTH // MAP_SCALE, HEIGHT // MAP_SCALE))
        self.clock = pygame.time.Clock()
        self.drawing = Drawing(self.sc, self.sc_map)

        self.road_coords = set()
        self.finish_coords = set()
        self.wall_coord_list = list()
        self.world_map, self.collision_walls = get_map(TILE)
        for coord, signature in self.world_map.items():
            if signature == "1":
                self.wall_coord_list.append(coord)
            elif signature == "2":
                self.finish_coords.add(coord)
            elif signature == ".":
                self.road_coords.add(coord)

    def player_setup(self):
        # self.loading()

        self.player = Player(self.sc, self.collision_walls, self.finish_coords)

        self.player.color = (self.color_r, self.color_g, self.color_b)
        self.player.init_angle = math.pi + (math.pi/2)
        self.player.rays = RayCast(self.world_map)

        with cbook.get_sample_data(PDF_LOAD_PATH) as image_file:
            _weights = plt.imread(image_file)
        
        weights_ids = np.rot90(_weights, axes=(2,0))

        val = np.array([-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0])

        weights = (
            val[np.around(weights_ids[0, :5, :] * 8).astype(np.uint8)], 
            val[np.around(weights_ids[1] * 8).astype(np.uint8)], 
            val[np.around(weights_ids[2, :, :3] * 8).astype(np.uint8)]
        )
        
        # self.player.brain = g.Ganglion(self.loaded_parameters[0], self.loaded_parameters[1], self.loaded_parameters[2], self.loaded_parameters[3])
        # self.player.brain = nn.Ganglion_numpy(self.loaded_parameters[2], self.loaded_indices, self.loaded_weights)
        self.player.brain = nn.Ganglion_numpy(weights)
        self.player.test_mode = True

        self.player.setup()

    def game_event(self):
        # drawing.background()

        self.player.movement()
        self.player.draw()
        
        for x, y in self.wall_coord_list:
            pygame.draw.rect(self.sc, WALL_COLOR_1, (x, y, TILE, TILE), 2)
        for x, y in self.finish_coords:
            pygame.draw.rect(self.sc, WALL_COLOR_2, (x, y, TILE, TILE), 2)

        self.drawing.info(0, 0, 1, self.clock)

    def run(self):
        self.player_setup()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
            self.sc.fill(BLACK)

            self.game_event()

            pygame.display.flip()
            self.clock.tick()


if __name__ == "__main__":
    game = Game()
    game.run()