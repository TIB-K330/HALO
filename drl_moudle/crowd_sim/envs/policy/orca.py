import numpy as np
import rvo2
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import ObservableState, JointState_2tyeps, JointState
# from crowd_nav.utils.a_star import Astar
from crowd_sim.envs.policy.subtarget import subtarget

def rotate_2d_point(theta, point):
    cos = np.cos(theta)
    sin = np.sin(theta)
    rotated_point_x = point[0] * cos - point[1] * sin
    rotated_point_y = point[0] * sin + point[1] * cos
    return (rotated_point_x, rotated_point_y)

def rotate_2d_points(theta, points):
    rotated_2d_points = []
    for point in points:
        rotated_2d_point = rotate_2d_point(theta, point)
        rotated_2d_points.append(rotated_2d_point)
    return rotated_2d_points

class ORCA(Policy):
    def __init__(self):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super().__init__()
        self.name = 'ORCA'
        self.trainable = False
        self.multiagent_training = True
        self.kinematics = 'holonomic'
        self.safety_space = 0
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.radius = 0.3
        self.max_speed = 1
        self.sim = None
        self.speed_samples = 5
        self.rotation_samples = 16
        self.sampling = None
        self.rotation_constraint = None
        # self.astarplanner = Astar()
        self.target_pos = None

    def configure(self, config, device='cpu'):
        self.set_common_parameters(config)
        return

    def set_common_parameters(self, config):
        self.kinematics = config.action_space.kinematics
        self.sampling = config.action_space.sampling
        self.speed_samples = config.action_space.speed_samples
        self.rotation_samples = config.action_space.rotation_samples
        self.rotation_constraint = config.action_space.rotation_constraint

    def set_phase(self, phase):
        return

    def predict(self, state):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        # state = self.state_transform(state)
        robot_state = state.robot_state
        cur_theta = robot_state.theta
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            robot_no = self.sim.addAgent(robot_state.position, *params, robot_state.radius + 0.1, # + self.safety_space,
                              robot_state.v_pref, robot_state.velocity)
            AHVVertices = [(-0.2, -0.12), (0.4, -0.4), (0.97, -0.26), (0.97, 0.26), (0.4, 0.4), (-0.2, 0.12)]
            rotated_AHVVertices = rotate_2d_points(robot_state.theta, AHVVertices)
            vertices_no = self.sim.setAHVConstraint(robot_no, rotated_AHVVertices)
            # print(vertices_no)
            for human_state in state.human_states:
                self.sim.addAgent(human_state.position, *params, human_state.radius + self.safety_space,
                                  self.max_speed, human_state.velocity)
            for wall in state.wall_states:
                self.sim.addObstacle([(wall.sx, wall.sy), (wall.ex, wall.ey)])
                self.sim.processObstacles()
            obstacle_params = 0.0, 0.0, 0.0, 0.0
            for obstacle in state.obstacle_states:
                self.sim.addAgent((obstacle.px, obstacle.py), *obstacle_params, obstacle.radius, 0.0, (0.0, 0.0))
        else:
            self.sim.setAgentPosition(0, robot_state.position)
            self.sim.setAgentVelocity(0, robot_state.velocity)
            AHVVertices = [(-0.2, -0.12), (0.4, -0.4), (0.97, -0.26), (0.97, 0.26), (0.4, 0.4), (-0.2, 0.12)]
            rotated_AHVVertices = rotate_2d_points(robot_state.theta, AHVVertices)
            vertices_no = self.sim.setAHVConstraint(0, rotated_AHVVertices)
            for human_state in state.human_states:
                self.sim.addAgent(human_state.position, *params, human_state.radius + self.safety_space,
                                  self.max_speed, human_state.velocity)
            for wall in state.wall_states:
                self.sim.addObstacle([(wall.sx, wall.sy), (wall.ex, wall.ey)])
                self.sim.processObstacles()
            obstacle_params = 0.0, 0.0, 0.0, 0.0
            for obstacle in state.obstacle_states:
                self.sim.addAgent((obstacle.px, obstacle.py), *obstacle_params, obstacle.radius, 0.0, (0.0, 0.0))
            # print(vertices_no)

            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.

        dis = np.sqrt((robot_state.gx - robot_state.px) * (robot_state.gx - robot_state.px) + (
                    robot_state.gy - robot_state.py) * (robot_state.gy - robot_state.py))
        if dis < 2.0:
            velocity = np.array((robot_state.gx - robot_state.px, robot_state.gy - robot_state.py))
        else:
            target_theta = subtarget(robot_state,state.human_states, state.obstacle_states, state.wall_states)
            velocity = np.array((np.cos(target_theta), np.sin(target_theta)))
            # target_pos = self.astarplanner.set_state2((robot_state, state.human_states, state.obstacle_states, state.wall_states))
            # if target_pos is None:
            #     # print('no solution')
            #     target_pos = (robot_state.gx, robot_state.px)
            # dis = (target_pos[0] - robot_state.px) * (target_pos[0] - robot_state.px) + (target_pos[1] - robot_state.py) * (target_pos[1] - robot_state.py)
            # dis = np.sqrt(dis)
            # velocity = np.array((target_pos[0] - robot_state.px, target_pos[1] - robot_state.py)) / dis
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(state.human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        speed_samples = self.speed_samples
        rotation_samples = self.rotation_samples
        d_vel = 1.0 / speed_samples / 2
        if self.kinematics == 'holonomic':
            action = ActionXY(*self.sim.getAgentVelocity(0))
        elif self.kinematics == 'unicycle':
            action = ActionXY(*self.sim.getAgentVelocity(0))
            theta = np.arctan2(action.vy, action.vx)
            vel = np.sqrt(action.vx*action.vx + action.vy * action.vy)
            theta = (theta - cur_theta + np.pi) % (2 * np.pi) - np.pi
            action = ActionRot(vel, theta)
        # vel_index = (np.sqrt(action.vx * action.vx + action.vy * action.vy) // d_vel + 1) // 2
        # if vel_index > speed_samples:
        #     vel_index = speed_samples
        # d_rot = np.pi / rotation_samples
        # rot_index = (np.arctan2(action.vy, action.vx) // d_rot + 1) // 2
        # if rot_index < 0:
        #     rot_index = rot_index + rotation_samples
        # if vel_index == 0:
        #     action_index = int(0)
        # else:
        #     action_index = int((vel_index - 1) * rotation_samples + rot_index + 1)
        # action = ActionXY(vel_index * d_vel * 2.0 * np.cos(rot_index * d_rot * 2.0), vel_index * d_vel * 2.0 *
        #                   np.sin(rot_index * d_rot * 2.0))
        action_index = -1
        self.last_state = state

        return action, action_index

    def state_transform(self,state):
        human_state = state.human_states
        obstacle_state = state.obstacle_states
        robot_state = state.robot_state
        wall_state = state.wall_state
        obs_state = [human_state, obstacle_state, wall_state]
        # for i in range((len(obstacle_state))):
        #     obstacle_human = ObservableState(obstacle_state[i].px, obstacle_state[i].py, 0.0, 0.0, obstacle_state[i].radius)
        #     human_state.append(obstacle_human)
        state = JointState(robot_state, obs_state)
        return state


class CentralizedORCA(ORCA):
    def __init__(self):
        super().__init__()
        self.name = 'ORCA'
        self.kinematics = 'holonomic'
        self.safety_space = 0
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.radius = 0.3
        self.max_speed = 1
        self.sim = None
        self.speed_samples = 5
        self.rotation_samples = 16
        self.sampling = None
        self.rotation_constraint = None
        self.time_step = 0.25
        self.obstacles = None
        self.walls = None
        self.wall_update_flag = False
        self.obstacle_update_flag = False

    def reset(self):
        del self.sim
        self.sim = None
        self.wall_update_flag = True
        self.obstacle_update_flag = True

    def set_static_obstacles(self, obstacles):
        self.obstacles = obstacles

    def set_walls(self, walls):
        self.walls = walls

    def predict(self, state):
        """ Centralized planning for all agents """
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

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        for i, agent_state in enumerate(state):
            velocity = np.array((agent_state.gx - agent_state.px, agent_state.gy - agent_state.py))
            speed = np.linalg.norm(velocity)
            pref_vel = velocity / speed if speed > 1 else velocity
            self.sim.setAgentPrefVelocity(i, (pref_vel[0], pref_vel[1]))

        if self.wall_update_flag and self.walls is not None:
            for i in range(len(self.walls)):
                wall = self.walls[i]
                self.sim.addObstacle([(wall.sx, wall.sy), (wall.ex, wall.ey)])
                self.sim.processObstacles()

            self.wall_update_flag = False

        if self.obstacle_update_flag and self.obstacles is not None:
            obstacle_params = 0.0, 0.0, 0.0, 0.0
            for i in range(len(self.obstacles)):
                obstacle = self.obstacles[i]
                self.sim.addAgent((obstacle.px, obstacle.py), *obstacle_params, obstacle.radius, 0.0, (0.0, 0.0))

            self.obstacle_update_flag = False

        self.sim.doStep()
        # actions = [ActionXY(0.0, 0.0) for i in range(len(state))]
        actions = [ActionXY(*self.sim.getAgentVelocity(i)) for i in range(len(state))]

        return actions
