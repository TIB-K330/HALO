import numpy as np
from numpy.linalg import norm
from matplotlib import animation, patches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import gym

from typing import Tuple

from simple_env.agent import Human, Obstacle, Wall
from simple_env.info import *
from simple_env.orca import CentralizedORCA


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, is_action_mask: bool = False):
        self.time_limit = 30
        self.time_step = 0.2
        self.global_time = None
        self.robot = None
        self.obstacles = None
        self.walls = None
        self.poly_obstacles = []

        self.discomfort_dist = 0.2

        # simulation configuration
        self.square_width = 10
        self.circle_radius = 4
        self.human_num = 5
        self.static_obstacle_num = 3

        self.centralized_planner = CentralizedORCA()

        # for visualization    
        self.panel_width = 10
        self.panel_height = 10
        self.states = None
        self.pred_trajs = None
        self.network_outputs = None
        self.action_mask_viss = None
        self.action_mask_values = None
        self.is_action_mask = is_action_mask


    def set_robot(self, robot):
        self.robot = robot

    def set_human_number(self, human_num):
        self.human_num = human_num
    def generate_human(self, human=None, non_stop=False, square=False):
        if human is None:
            human = Human()

        if square is False and non_stop is False:
            for sample_count in range(200):
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = (np.random.random() - 0.5) * human.v_pref
                py_noise = (np.random.random() - 0.5) * human.v_pref
                gx_noise = (np.random.random() - 0.5) * human.v_pref
                gy_noise = (np.random.random() - 0.5) * human.v_pref
                
                px = (self.circle_radius + 0.5) * np.cos(angle) + px_noise
                py = (self.circle_radius + 0.5) * np.sin(angle) + py_noise
                gx = -(self.circle_radius + 0.5) * np.cos(angle) + gx_noise
                gy = -(self.circle_radius + 0.5) * np.sin(angle) + gy_noise
                collide = False

                for agent in [self.robot]:
                    min_dist = human.radius + agent.radius + self.discomfort_dist + 0.5
                    if (norm((px - agent.px, py - agent.py)) < min_dist or
                        norm((px - agent.gx, py - agent.gy)) < min_dist):
                        collide = True
                        break
                
                for agent in self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if (norm((px - agent.px, py - agent.py)) < min_dist or
                        norm((px - agent.gx, py - agent.gy)) < min_dist):
                        collide = True
                        break
                
                for wall in self.walls:
                    p_dist = self.point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, px, py)
                    g_dist = self.point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, gx, gy)
                    if (p_dist < human.radius + 0.3 or g_dist < human.radius + 0.3):
                        collide = True
                        break
                
                for poly in self.poly_obstacles:
                    if self.point_in_poly(px, py, poly) or self.point_in_poly(gx, gy, poly):
                        collide = True
                        break

                if not collide:
                    break
            
            human.start_pos.append((px, py))
            human.set(px, py, gx, gy, 0, 0)
        elif square is False and non_stop is True:
            for sample_count in range(200):
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px = human.px
                py = human.py
                gx_noise = (np.random.random() - 0.5) * human.v_pref
                gy_noise = (np.random.random() - 0.5) * human.v_pref
                gx = self.circle_radius * np.cos(angle) + gx_noise
                gy = self.circle_radius * np.sin(angle) + gy_noise
                collide = False

                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist + 0.5
                    if norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                        collide = True
                        break
                
                for agent in [self.robot]:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                        collide = True
                        break

                for wall in self.walls:
                    g_dist = self.point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, gx, gy)
                    if g_dist < human.radius + self.robot.radius:
                        collide = True
                        break

                for poly in self.poly_obstacles:
                    if self.point_in_poly(gx, gy, poly):
                        collide = True
                        break

                if not collide:
                    break

            human.start_pos.append((px, py))
            human.set(px, py, gx, gy, 0, 0)
        elif square is True and non_stop is False:
            pass

        elif square is True and non_stop is True:
            pass

        return human

    def generate_static_obstacle(self):
        self.obstacles = []
        # obst_1 = Obstacle()
        # obst_1.set(0.0, -2.5)
        # self.obstacles.append(obst_1)

        # obst_2 = Obstacle()
        # obst_2.set(2.0, 0.0)
        # self.obstacles.append(obst_2)

        # obst_3 = Obstacle()
        # obst_3.set(2.0, -1.5)
        # self.obstacles.append(obst_3)
        for i in range(self.static_obstacle_num):
            obstacle = Obstacle()
            obstacle.sample_random_attributes()
            
            for sample_count in range(200):
                px = (np.random.random() - 0.5) * self.square_width * 0.8
                py = (np.random.random() - 0.5) * self.circle_radius * 2
                obstacle.set(px, py, obstacle.radius)
                collide = False

                for agent in [self.robot] + self.humans:
                    if (norm((px - agent.px, py - agent.py)) < obstacle.radius + agent.radius + 0.5 or
                        norm((px - agent.gx, py - agent.gy)) < obstacle.radius + agent.radius + 0.5):
                        collide = True
                        break

                for agent in self.obstacles:
                    if norm((px - agent.px, py - agent.py)) < obstacle.radius + agent.radius + 0.5:
                        collide = True
                        break

                for wall in self.walls:
                    if self.point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, px, py) < obstacle.radius + 0.8:
                        collide = True
                        break
                
                for poly in self.poly_obstacles:
                    if self.point_in_poly(px, py, poly):
                        collide = True
                        break

                if not collide:
                    break

            if sample_count < 200:
                self.obstacles.append(obstacle)

    def generate_wall(self, start_position, end_position):
        wall = Wall()
        wall.set_position(start_position, end_position)
        return wall

    def generate_center_obstacle(self):
        center_x = (np.random.random() - 0.5) * 2  # gen random num in [-1, 1]
        center_y = (np.random.random() - 0.5) * 2

        # the width and length of the rect rand in [1, 3]
        width = np.clip(np.random.normal(2, 1.0), 1, 3)
        length = np.clip(np.random.normal(2, 1.0), 1, 3)

        x1 = center_x - width / 2.0
        x2 = center_x + width / 2.0
        y1 = center_y - length / 2.0
        y2 = center_y + length / 2.0

        transfer_vertex =([x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1])
        
        # gen walls based on the vertex of the rect
        for i in range(len(transfer_vertex) - 1):
            self.walls.append(self.generate_wall(transfer_vertex[i], transfer_vertex[i + 1]))

        self.poly_obstacles.clear()
        self.poly_obstacles.append(transfer_vertex)

    def generate_corridor_scenario(self):
        corridor_width = self.square_width
        corridor_length = self.square_width * 2.0  # as a large num
        
        self.walls.append(self.generate_wall([-corridor_width / 2, corridor_length / 2], 
                                             [-corridor_width / 2, -corridor_length / 2]))

        self.walls.append(self.generate_wall([corridor_width / 2, -corridor_length / 2], 
                                             [corridor_width / 2, corridor_length / 2]))

    def reset(self, test_case=None):
        if self.robot is None:
            raise AttributeError('Robot has to be set!')
        
        self.global_time = 0
        target_x = (np.random.random() - 0.5) * self.square_width * 0.8  # rand in [-4, 4]
        target_y = self.circle_radius  # fixed to 4.0
        robot_theta = np.pi / 2 + np.random.random() * np.pi / 4.0 - np.pi / 8.0

        # self.robot.set(0.0, -4.0, 3.0, 4.0, 0.0, 0.0, np.pi / 2, 0.3, 1.0)
        self.robot.set(0.0, -self.circle_radius, target_x, target_y, 0.0, 0.0, robot_theta, 0.3, 1.0)

        self.walls = []
        self.generate_corridor_scenario()
        self.generate_center_obstacle()

        self.humans = []
        for _ in range(self.human_num):
            self.humans.append(self.generate_human())

        self.generate_static_obstacle()

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step

        self.centralized_planner.time_step = self.time_step
        self.centralized_planner.reset()
        
        self.states = list()
        if hasattr(self.robot.policy, 'get_pred_traj'):
            self.pred_trajs = list()
            self.network_outputs = list()
        if self.is_action_mask:
            if hasattr(self.robot.policy, 'get_action_mask_result'):
                self.action_mask_viss = list()
                self.action_mask_values = list()

        ob_humans = []
        for human in self.humans:
            ob_humans.append(human.get_observable_state())

        if self.obstacles is not None:
            ob_obstacles = [obstacle.get_observable_state() for obstacle in self.obstacles]
        else:
            ob_obstacles = None

        if self.walls is not None:
            ob_walls = [wall.get_observable_state() for wall in self.walls]
        else:
            ob_walls = None

        return (ob_humans, ob_obstacles, ob_walls)

    def reward_cal(self, action, human_actions=None):
        collision = False
        pre_robot_pos_x, pre_robot_pos_y = self.robot.compute_position(action, self.time_step)

        """Secondly, deal with humans """
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            pre_hum_pos_x = human_actions[i].vx * self.time_step + human.px
            pre_hum_pos_y = human_actions[i].vy * self.time_step + human.py
            ex = pre_hum_pos_x - pre_robot_pos_x
            ey = pre_hum_pos_y - pre_robot_pos_y
            closest_dist = self.point_to_segment_dist(px, py, ex, ey, 0, 0)

            if closest_dist < human.radius + self.robot.radius:
                collision = True
                print("Collision: distance between robot and human %d is %f at time %f"%(i, closest_dist, 
                                                                                         self.global_time))

        """ Thirdly, deal with obstacles"""
        for i, obstacle in enumerate(self.obstacles):
            closest_dist = self.point_to_segment_dist(self.robot.px, self.robot.py, 
                                                      pre_robot_pos_x, pre_robot_pos_y, 
                                                      obstacle.px, obstacle.py)
            
            if closest_dist < obstacle.radius + self.robot.radius:
                collision = True
                print("Collision: distance between robot and obstacle %d is %f at time %f"%(i, closest_dist,
                                                                                            self.global_time))

        """ Then, deal with walls"""
        for i, wall in enumerate(self.walls):
            cur_at_left = self.counterclockwise(wall.sx, wall.sy, wall.ex, wall.ey, self.robot.px, self.robot.py)
            pre_at_left = self.counterclockwise(wall.sx, wall.sy, wall.ex, wall.ey, pre_robot_pos_x, pre_robot_pos_y)
            
            wall_sp_at_left = self.counterclockwise(self.robot.px, self.robot.py, pre_robot_pos_x, pre_robot_pos_y, 
                                                    wall.sx, wall.sy)
            
            wall_ep_at_left = self.counterclockwise(self.robot.px, self.robot.py, pre_robot_pos_x, pre_robot_pos_y, 
                                                    wall.ex, wall.ey)
            
            if cur_at_left != pre_at_left and wall_sp_at_left != wall_ep_at_left:
                closest_dist = 0.0  # across the wall
            else:
                min_dis_start = self.point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, self.robot.px, 
                                                           self.robot.py)
                
                min_dis_end = self.point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, pre_robot_pos_x, 
                                                         pre_robot_pos_y)
                
                closest_dist = min_dis_end if min_dis_end < min_dis_start else min_dis_start

            if closest_dist < self.robot.radius:
                collision = True
                print("Collision: distance between robot and wall %d is %f at time %f"%(i, closest_dist,
                                                                                        self.global_time))


        """check if reaching the goal"""
        pred_position = np.array([pre_robot_pos_x, pre_robot_pos_y])
        goal_position = np.array(self.robot.get_goal_position())
        reaching_goal = norm(pred_position - goal_position) < self.robot.radius

        if self.global_time >= self.time_limit - 1:
            done = True
            info = Timeout()
        elif collision:
            done = True
            info = Collision()
        elif reaching_goal:
            done = True
            info = ReachGoal()
        else:
            done = False
            info = Nothing()

        return done, info

    def step(self, action) -> Tuple[Tuple, bool, Timeout | Collision | ReachGoal | Nothing]:
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        agent_states = [human.get_full_state() for human in self.humans]
        self.centralized_planner.set_walls(self.walls)
        self.centralized_planner.set_static_obstacles(self.obstacles)
        human_actions = self.centralized_planner.predict(agent_states)
        done, info = self.reward_cal(action, human_actions)

        if hasattr(self.robot.policy, 'get_pred_traj'):
            # for debug and visualization
            pred_traj, local_goal = self.robot.policy.get_pred_traj()
            self.pred_trajs.append(pred_traj)
            self.network_outputs.append(local_goal)

        if self.is_action_mask:
            if hasattr(self.robot.policy, 'get_action_mask_result'):
                action_mask_vis, action_mask = self.robot.policy.get_action_mask_result()
                self.action_mask_viss.append(action_mask_vis)
                self.action_mask_values.append(action_mask)


        self.robot.step(action)
        
        for human, action in zip(self.humans, human_actions):
            human.step(action)
            if not human.reached_destination():
                continue
            
            human.reach_count += 1
            if human.reach_count <= 2:
                continue

            if norm((human.px - self.robot.px, human.py - self.robot.py)) > human.radius + self.robot.radius + 0.5:
                self.generate_human(human, non_stop=True)
                human.reach_count = 0
        
        self.global_time += self.time_step
        self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans], 
                            [human.id for human in self.humans]])
        
        ob_humans = [human.get_observable_state() for human in self.humans]
        ob_obstacles = [obstacle.get_observable_state() for obstacle in self.obstacles]
        ob_walls = [wall.get_observable_state() for wall in self.walls]

        ob = (ob_humans, ob_obstacles, ob_walls)
        return ob, done, info

    def render(self):
        x_offset = 0.3
        y_offset = 0.4
        cmap = plt.cm.get_cmap('terrain', 200)
        arrow_style = patches.ArrowStyle.Fancy(head_length=4, head_width=2, tail_width=.6)
        
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.tick_params(labelsize=12)
        ax.set_xlim(-self.panel_width / 2 - 1, self.panel_width / 2 + 1)
        ax.set_ylim(-self.panel_height / 2 - 0.5, self.panel_height / 2 + 0.5)
        ax.set_xlabel('x(m)', fontsize=14)
        ax.set_ylabel('y(m)', fontsize=14)

        robot_color = 'black'
        human_colors = [cmap(20) for i in range(len(self.humans))]
        robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                    color=robot_color, marker='o', linestyle='None', markersize=15)
        
        ax.add_artist(robot_start)
        robot_positions = [state[0].position for state in self.states]
        robot = plt.Circle(robot_positions[0], self.robot.radius, fill=False, color=robot_color)
        ax.add_artist(robot)

        robot_goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                    color='red', marker='*', linestyle='None', markersize=15, label='Goal')

        ax.add_artist(robot_goal)

        # add static circular obstacles
        for i in range(len(self.obstacles)):
            obstacle = self.obstacles[i]
            obstacle_mark = plt.Circle(obstacle.get_position(), obstacle.radius, fill=True, color='grey')
            ax.add_artist(obstacle_mark)

        # add static polygon obstacles
        for i in range(len(self.walls)):
            wall = self.walls[i]
            wall_line = mlines.Line2D([wall.sx, wall.ex], [wall.sy, wall.ey], color='black', marker='.', 
                                      linestyle='solid', markersize=5)
            
            ax.add_artist(wall_line)

        direction_length = 1.0
        if len(self.humans) == 0:
            pass 
        else:
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False, color=human_colors[i])
                      for i in range(len(self.humans))]
            
            plt.legend([robot, humans[0], robot_goal], ['Robot', 'Human', 'Goal'], fontsize=14)

            human_numbers = []
            for i in range(len(self.humans)):
                human_numbers.append(plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset,
                                              str(i + 1), color='black', fontsize=12))

            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

        time = plt.text(0.4, 1.02, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
        ax.add_artist(time)

        radius = self.robot.radius
        orientations = []
        
        for i in range(self.human_num + 1):
            orientation = []
            
            for state in self.states:
                if i == 0:
                    agent_state = state[0]
                    direction = ((agent_state.px, agent_state.py),
                                 (agent_state.px + direction_length * radius * np.cos(agent_state.theta),
                                  agent_state.py + direction_length * radius * np.sin(agent_state.theta)))
                else:
                    agent_state = state[1][i - 1]
                    # assume that the heading of holonomic agents is inconsistent with their velocity vector
                    theta = np.arctan2(agent_state.vy, agent_state.vx)
                    direction = ((agent_state.px, agent_state.py),
                                 (agent_state.px + direction_length * radius * np.cos(theta),
                                  agent_state.py + direction_length * radius * np.sin(theta)))
            
                orientation.append(direction)
            
            orientations.append(orientation)

            if i == 0:
                robot_arrow_color = 'red'
                arrows = [patches.FancyArrowPatch(*orientation[0], color=robot_arrow_color, arrowstyle=arrow_style)]
            else:
                human_arrow_color = 'red'
                arrows.extend([patches.FancyArrowPatch(*orientation[0], color=human_arrow_color, 
                                                       arrowstyle=arrow_style)])
                
        for arrow in arrows:
            ax.add_artist(arrow)            
        
        global_step = 0
        
        def update(frame_num):
            nonlocal global_step
            nonlocal arrows

            global_step = frame_num
            robot.center = robot_positions[frame_num]  # update the position of robot in the animation

            if self.human_num > 0:
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]  # update the position of humans
                    human_numbers[i].set_position((human.center[0], human.center[1]))  # update the position of human numbers

            for arrow in arrows:
                arrow.remove()

            for i in range(self.human_num + 1):
                orientation = orientations[i]
                if i == 0:
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=robot_arrow_color, 
                                                      arrowstyle=arrow_style)]
                else:
                    arrows.extend([patches.FancyArrowPatch(*orientation[frame_num], color=human_arrow_color, 
                                                           arrowstyle=arrow_style)])
            
            for arrow in arrows:
                ax.add_artist(arrow)           

            plot_opt_traj()
            time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

        vis_traj = None
        vis_local_goal = None
        vis_action_mask = None
        def plot_opt_traj():
            nonlocal vis_traj, vis_local_goal, vis_action_mask

            if vis_traj != None:
                vis_traj.remove()

            if vis_local_goal != None:
                vis_local_goal.remove()
            
            if vis_action_mask != None:
                for handle in vis_action_mask:
                    handle.remove()
            vis_action_mask = []

            opt_traj = self.pred_trajs[global_step]
            vis_traj = mlines.Line2D(opt_traj[:, 0], opt_traj[:, 1])
            ax.add_artist(vis_traj)

            local_goal = self.network_outputs[global_step]
            vis_local_goal = mlines.Line2D([local_goal[0]], [local_goal[1]], color='blue', marker='*', 
                                           linestyle='None', markersize=12, label='local_goal')

            ax.add_artist(vis_local_goal)
            # if self.is_action_mask:
            #     action_mask_vis = self.action_mask_viss[global_step]
            #     action_mask = self.action_mask_values[global_step]
            #     for candidate_goal, mask_value in zip(action_mask_vis, action_mask):
            #         if mask_value == 0.0:
            #             handle = ax.plot(candidate_goal[0], candidate_goal[1], 'ro')[0]
            #         elif mask_value == 2.0:
            #             handle = ax.plot(candidate_goal[0], candidate_goal[1], 'bo')[0]
            #         else:
            #             # print(mask_value)
            #             handle = ax.plot(candidate_goal[0], candidate_goal[1], 'go')[0]
            #         vis_action_mask.append(handle)



        anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
        anim.running = True
        plt.show()
        # anim.save('animation.gif', writer='imagemagick')

    @staticmethod
    def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
        """
        Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

        """
        dx = x2 - x1;
        dy = y2 - y1;

        if norm((dx, dy)) < 1e-9:
            return norm((x3 - x1, y3 - y1))

        u = ((x3 - x1) * dx + (y3 - y1) * dy) / (dx * dx + dy * dy)
        
        # distance to the seg not the line
        if u > 1: 
            u = 1
        elif u < 0:
            u = 0
        
        # (x, y) is the closest point to (x3, y3) on the line segment
        x = x1 + u * dx
        y = y1 + u * dy

        return norm((x - x3, y - y3))

    @staticmethod
    def counterclockwise(x1, y1, x2, y2, x3, y3):
        """
        Calculate if  point (x3, y3) lies in the left side of directed line segment from (x1, y1) to (x2, y2)

        """
        vec1 = np.array([x2 - x1, y2 - y1])
        vec2 = np.array([x3 - x1, y3 - y1])
        return True if np.cross(vec1, vec2) > 0 else False

    def point_in_poly(self, px, py, vertex):
        """
        Calculate if  point (px, py) lies in the polygons represented by vertex (counterclockwise)
        """
        for i in range(len(vertex) - 1):
            p1_x = vertex[i][0]
            p1_y = vertex[i][1]
            p2_x = vertex[i + 1][0]
            p2_y = vertex[i + 1][1]

            if not self.counterclockwise(p1_x, p1_y, p2_x, p2_y, px, py):
                return False
            
        return True