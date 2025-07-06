import abc
import logging
import numpy as np
from crowd_sim.envs.utils.state import ObstacleState


class Obstacle(object):
    def __init__(self):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.radius = 0.3
        self.px = None
        self.py = None

    # def __init__(self, px, py):
    #     """
    #     Base class for robot and human. Have the physical attributes of an agent.
    #
    #     """
    #     self.radius = 0.3
    #     self.px = px
    #     self.py = py

    def print_info(self):
        logging.info('obstacle locates at {} and {}'.format(
             self.px, self.py))

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.radius = np.clip(np.random.normal(0.25, 0.05), 0.1, 0.4)


    def set(self, px, py, radius=None):
        self.px = px
        self.py = py
        if radius is not None:
            self.radius = radius

    def get_observable_state(self):
        return ObstacleState(self.px, self.py, self.radius)

    def get_full_state(self):
        return self.get_observable_state()

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]



