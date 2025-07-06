import rvo2
import numpy as np
from abc import ABC
from collections import namedtuple

ActionXY = namedtuple('ActionXY', ['vx', 'vy'])

class CentralizedORCA(ABC):
    def __init__(self):
        super().__init__()
        self.safety_space = 0
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.radius = 0.3
        self.max_speed = 1.0
        self.sim = None
        self.time_step = 0.2
        self.obstacles = None
        self.walls = None
        self.wall_update_flag = False
        self.obst_update_flag = False

    def reset(self):
        del self.sim
        self.sim = None
        self.wall_update_flag = True
        self.obst_update_flag = True

    def set_static_obstacles(self, obstacles):
        self.obstacles = obstacles

    def set_walls(self, walls):
        self.walls = walls

    def predict(self, state):
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and self.sim.getNumAgents() != len(state):
            self.reset()

        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            for agent_state in state:
                self.sim.addAgent(agent_state.position, *params, agent_state.radius + 0.01 + self.safety_space, 
                                  self.max_speed, agent_state.velocity)
        else:
            for i, agent_state in enumerate(state):
                self.sim.setAgentPosition(i, agent_state.position)
                self.sim.setAgentVelocity(i, agent_state.velocity)

        for i, agent_state in enumerate(state):
            velocity = np.array((agent_state.gx - agent_state.px, agent_state.gy - agent_state.py))
            speed = np.linalg.norm(velocity)
            pref_vel = velocity / speed if speed > self.max_speed else velocity
            self.sim.setAgentPrefVelocity(i, (pref_vel[0], pref_vel[1]))

        if self.wall_update_flag and self.walls:
            for i in range(len(self.walls)):
                wall = self.walls[i]
                self.sim.addObstacle([(wall.sx, wall.sy), (wall.ex, wall.ey)])
                self.sim.processObstacles()
            
            self.wall_update_flag = False

        if self.obst_update_flag and self.obstacles:
            obstacle_params = 0.0, 0.0, 0.0, 0.0
            for i in range(len(self.obstacles)):
                obstacle = self.obstacles[i]
                self.sim.addAgent((obstacle.px, obstacle.py), *obstacle_params, obstacle.radius, 0.0, (0.0, 0.0))

            self.obst_update_flag = False
        
        self.sim.doStep()

        actions = [ActionXY(*self.sim.getAgentVelocity(i)) for i in range(len(state))]
        return actions