import abc
import logging
import numpy as np
from crowd_sim.envs.utils.state import WallState


class Wall(object):
    def __init__(self, config):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.sx = None
        self.sy = None
        self.ex = None
        self.ey = None

    def set(self, sx, sy, ex, ey):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey

    def print_info(self):
        logging.info('wall from {}, {} to {}, {}'.format(
             self.sx, self.sy, self.ex, self.ey))

    def get_observable_state(self):
        return WallState(self.sx, self.sy, self.ex, self.ey)

    def get_full_state(self):
        return self.get_observable_state()

    def get_position(self):
        return self.sx, self.sy, self.ex, self.ey

    def set_position(self, start_position, end_position):
        self.sx = start_position[0]
        self.sy = start_position[1]
        self.ex = end_position[0]
        self.ey = end_position[1]


