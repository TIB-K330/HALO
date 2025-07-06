from typing import Optional, Tuple
import dgl
import torch
import numpy as np
from .rvo.rvo_inter import rvo_inter

JointStateTensor = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class CrowdNavGraph:
    """
    ntypes = ['robot', 'human', 'obstacle', 'wall'],
    etypes = ['h2r', 'o2r', 'w2r', 'h2h', 'o2h', 'w2h', 'r2h']
    The pointed ntypes actively avoids obstacles, and those without active obstacle avoidance ability \
        will not be pointed, if the robot is invisible to human (default setting), etype 'r2h' will be removed
    """
    def __init__(self, state):
        super(CrowdNavGraph, self).__init__()
        self.graph = None  # type: Optional[dgl.DGLHeteroGraph]
        self.state = None  # type: Optional[JointStateTensor]
        self.rels = ['h2r', 'o2r', 'w2r', 'o2h', 'w2h', 'h2h']

        self.rvo_inter = rvo_inter(
            neighbor_region=6, 
            neighbor_num=20, 
            vxmax=1, 
            vymax=1, 
            acceler=1.0,
            env_train=True,
            exp_radius=0.0, 
            ctime_threshold=3.0, 
            ctime_line_threshold=3.0
        )
        rotated_rvo_state = self.config_rvo_state(state)
        self.build_up_graph_on_rvostate(rotated_rvo_state)

    def config_rvo_state(self, state: JointStateTensor) -> JointStateTensor:
        """
        Transform the features in map coordinate to rvo-based representation in agent-centric coordinate.
        Input tuple include robot state (tensor) and observations (tensor).
        robot state tensor is of size (number, state_length)(for example 1*9)
        human state tensor is of size (number, state_length)(for example 5*5)
        obstacle state tensor is of size (number, state_length)(for example 3*3)
        wall state tensor is of size (number, state_length)(for example 4*4)
        """
        # for robot
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
        #  0     1      2     3      4        5     6      7         8
        # for human
        #  'px', 'py', 'vx', 'vy', 'radius'
        #  0     1      2     3      4
        # for obstacle
        # 'px', 'py', 'radius'
        #  0     1     2
        # for wall
        # 'sx', 'sy', 'ex', 'ey', radius
        #  0     1     2     3
        assert len(state[0].shape) == 2
        assert state[2] is not None
        robot_state = state[0]
        human_state = state[1]
        obstacle_state = state[2]
        wall_state = state[3]

        robot_state_array = robot_state.numpy()
        human_state_array = human_state.numpy()
        obstacle_state_array = obstacle_state.numpy()
        wall_state_array = wall_state.numpy()

        rvo_human_state, rvo_obstacle_state, rvo_wall_state, _, _, _ = self.rvo_inter.config_vo_inf(
            robot_state_array,
            human_state_array, 
            obstacle_state_array,
            wall_state_array,
        )

        rvo_human_state = torch.Tensor(rvo_human_state)
        rvo_obstacle_state = torch.Tensor(rvo_obstacle_state)
        rvo_wall_state = torch.Tensor(rvo_wall_state)

        return self.world2robotframe(robot_state, rvo_human_state, rvo_obstacle_state, rvo_wall_state)

    @staticmethod
    def world2robotframe(robot_state, human_state, obstacle_state, wall_state) -> JointStateTensor:
        # calculate distance and direction to goal 
        dx = robot_state[:, 5] - robot_state[:, 0]
        dy = robot_state[:, 6] - robot_state[:, 1]
        dx = dx.unsqueeze(1)
        dy = dy.unsqueeze(1)
        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        rot = torch.atan2(dy, dx)
        
        robot_velocities = robot_state[:, 2:4]
        v_pref = robot_state[:, 7].unsqueeze(1)
        cur_heading = (robot_state[:, 8].unsqueeze(1) - rot + np.pi) % (2 * np.pi) - np.pi
        new_robot_state = torch.cat((robot_velocities, dg, v_pref, cur_heading), dim=1)

        cos_rot = torch.cos(rot)
        sin_rot = torch.sin(rot)
        transform_matrix = torch.cat((cos_rot, -sin_rot, sin_rot, cos_rot), dim=0).reshape(2, 2)
        
        if human_state.shape[0] != 0:
            human_state = torch.index_select(human_state, 1, torch.tensor([8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 10]))
            temp = human_state[:, :8]
            temp = temp.reshape(human_state.shape[0], -1, 2)
            temp = torch.matmul(temp, transform_matrix)
            human_state[:, :8] = temp.reshape(human_state.shape[0], -1)
            human_state[:, -1] = human_state[:, -1] + 0.3
        
        if obstacle_state.shape[0] != 0:
            obstacle_state = torch.index_select(obstacle_state, 1, torch.tensor([8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 10]))
            temp = obstacle_state[:, :8]
            temp = temp.reshape(obstacle_state.shape[0], -1, 2)
            temp = torch.matmul(temp, transform_matrix)
            obstacle_state[:, :8] = temp.reshape(obstacle_state.shape[0], -1)
            obstacle_state[:, -1] = obstacle_state[:, -1] + 0.3
        
        if wall_state.shape[0] != 0:
            wall_state = torch.index_select(wall_state, 1, torch.tensor([8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7]))
            temp = wall_state[:, :10]
            temp = temp.reshape(wall_state.shape[0], -1, 2)
            temp = torch.matmul(temp, transform_matrix)
            wall_state[:, :10] = temp.reshape(wall_state.shape[0], -1)
            wall_state = torch.cat((wall_state, 0.3 * torch.ones((wall_state.shape[0], 1))), dim=1)

        return new_robot_state, human_state, obstacle_state, wall_state

    def build_up_graph_on_rvostate(self, rvo_state: JointStateTensor) -> None:
        # define feature indices
        node_types_one_hot = ['robot', 'human', 'obstacle', 'wall']

        robot_metric_features = ['rob_vel_l', 'rob_vel_r', 'dis2goal', 'rob_vel_pre', 'rob_ori']
        human_metric_features = [
            'human_vo_px', 
            'human_vo_py', 
            'human_vo_vl_x', 
            'human_v0_vl_y', 
            'human_vo_vr_x',
            'human_vo_vr_y', 
            'human_min_dis', 
            'human_exp_time', 
            'human_pos_x', 
            'human_pos_y',
            'human_radius'
        ]

        obstacle_metric_features = [
            'obs_vo_px', 
            'obs_vo_py', 
            'obs_vo_vl_x', 
            'obs_v0_vl_y', 
            'obs_vo_vr_x',
            'obs_vo_vr_y', 
            'obs_min_dis', 
            'obs_exp_time', 
            'obs_pos_x', 
            'obs_pos_y',
            'obs_radius'
        ]

        wall_metric_features = [
            'wall_vo_px', 
            'wall_vo_py', 
            'wall_vo_vl_x', 
            'wall_v0_vl_y', 
            'wall_vo_vr_x',
            'wall_vo_vr_y', 
            'wall_min_dis', 
            'wall_exp_time', 
            'wall_sx', 
            'wall_sy', 
            'wall_ex',
            'wall_ey', 
            'wall_radius'
        ]

        all_features = (node_types_one_hot + robot_metric_features + human_metric_features + 
                        obstacle_metric_features + wall_metric_features)

        # copy input data
        self.state = rvo_state
        robot_state, human_state, obstacle_state, wall_state = self.state
        feature_dimensions = len(all_features)

        robot_num = robot_state.shape[0]
        human_num = 0 if human_state is None else human_state.shape[0]
        obstacle_num = 0 if obstacle_state is None else obstacle_state.shape[0]
        wall_num = 0 if wall_state is None else wall_state.shape[0]

        total_node_num = robot_num + human_num + obstacle_num + wall_num

        # fill feature data into the hetero-graph
        # data of the robot
        robot_tensor = torch.zeros((robot_num, feature_dimensions))
        robot_tensor[0, all_features.index('robot')] = 1
        robot_tensor[0, all_features.index('rob_vel_l'):all_features.index("rob_ori") + 1] = robot_state[0]
        features = robot_tensor
        # data of the pedestrians
        if human_num > 0:
            human_tensor = torch.zeros((human_num, feature_dimensions))
            for i in range(human_num):
                human_tensor[i, all_features.index('human')] = 1
                human_tensor[i, all_features.index('human_vo_px'):all_features.index("human_radius") + 1] = \
                    human_state[i]

            features = torch.cat([features, human_tensor], dim=0)
        # data of the static circular obstacles
        if obstacle_num > 0:
            obstacle_tensor = torch.zeros((obstacle_num, feature_dimensions))
            for i in range(obstacle_num):
                obstacle_tensor[i, all_features.index('obstacle')] = 1
                obstacle_tensor[i, all_features.index('obs_vo_px'):all_features.index("obs_radius") + 1] = \
                    obstacle_state[i]

            features = torch.cat([features, obstacle_tensor], dim=0)
        # data of the walls
        if wall_num > 0:
            wall_tensor = torch.zeros((wall_num, feature_dimensions))
            for i in range(wall_num):
                wall_tensor[i, all_features.index('wall')] = 1
                wall_tensor[i, all_features.index('wall_vo_px'):all_features.index("wall_radius") + 1] = \
                    wall_state[i]
            
            features = torch.cat([features, wall_tensor], dim=0)

        # build up edges for the social graph
        src_id = torch.Tensor([])
        dst_id = torch.Tensor([])
        edge_types = torch.Tensor([])
        edge_norm = torch.Tensor([])

        # add obstacle_to_robot edges
        if obstacle_num > 0:
            src_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
            # for example, obstacle_num=3, robot_num=1, human_num=5, src_obstacle_id=tensor([6, 7, 8])
            o2r_robot_id = torch.zeros_like(src_obstacle_id)
            o2r_edge_types = torch.ones_like(o2r_robot_id) * torch.LongTensor([self.rels.index('o2r')])
            o2r_edge_norm = torch.ones_like(o2r_robot_id) * 1.0
            src_id = src_obstacle_id
            dst_id = o2r_robot_id
            edge_types = o2r_edge_types
            edge_norm = o2r_edge_norm

        # add human_to_robot edges
        if human_num > 0:
            src_human_id = torch.tensor(range(human_num)) + robot_num
            h2r_robot_id = torch.zeros_like(src_human_id)
            h2r_edge_types = torch.ones_like(h2r_robot_id) * torch.LongTensor([self.rels.index('h2r')])
            h2r_edge_norm = torch.ones_like(h2r_robot_id) * 1.0
            src_id = torch.cat([src_id, src_human_id], dim=0)
            dst_id = torch.cat([dst_id, h2r_robot_id], dim=0)
            edge_types = torch.cat([edge_types, h2r_edge_types], dim=0)
            edge_norm = torch.cat([edge_norm, h2r_edge_norm], dim=0)

        # add wall_to_robot edges
        if wall_num > 0:
            src_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
            w2r_robot_id = torch.zeros_like(src_wall_id)
            w2r_edge_types = torch.ones_like(w2r_robot_id) * torch.LongTensor([self.rels.index('w2r')])
            w2r_edge_norm = torch.ones_like(w2r_robot_id) * 1.0

            src_id = torch.cat([src_id, src_wall_id], dim=0)
            dst_id = torch.cat([dst_id, w2r_robot_id], dim=0)
            edge_types = torch.cat([edge_types, w2r_edge_types], dim=0)
            edge_norm = torch.cat([edge_norm, w2r_edge_norm], dim=0)

        for j in range(human_num):
            i = j + robot_num
            if obstacle_num > 0:
                # add obstacle_to_human edges
                o2h_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
                o2h_human_id = torch.ones_like(src_obstacle_id) * i
                o2h_edge_types = torch.ones_like(o2h_human_id) * torch.LongTensor([self.rels.index('o2h')])
                o2h_edge_norm = torch.ones_like(o2h_human_id) * 1.0
                src_id = torch.cat([src_id, o2h_obstacle_id], dim=0)
                dst_id = torch.cat([dst_id, o2h_human_id], dim=0)
                edge_types = torch.cat([edge_types, o2h_edge_types], dim=0)
                edge_norm = torch.cat([edge_norm, o2h_edge_norm], dim=0)

            if wall_num > 0:
                # add wall_to_human edges
                w2h_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
                w2h_human_id = torch.ones_like(src_wall_id) * i
                w2h_edge_types = torch.ones_like(w2h_human_id) * torch.LongTensor([self.rels.index('w2h')])
                w2h_edge_norm = torch.ones_like(w2h_human_id) * 1.0
                src_id = torch.cat([src_id, w2h_wall_id], dim=0)
                dst_id = torch.cat([dst_id, w2h_human_id], dim=0)
                edge_types = torch.cat([edge_types, w2h_edge_types], dim=0)
                edge_norm = torch.cat([edge_norm, w2h_edge_norm], dim=0)

        if human_num > 1:
            # add human_to_human edges
            temp_src_id = []
            temp_dst_id = []
            for i in range(human_num):
                for k in range(i + 1, human_num):
                    temp_src_id.append(i + robot_num)
                    temp_src_id.append(k + robot_num)
                    temp_dst_id.append(k + robot_num)
                    temp_dst_id.append(i + robot_num)

            h2h_src_id = torch.IntTensor(temp_src_id)
            h2h_dst_id = torch.IntTensor(temp_dst_id)
            h2h_edge_types = torch.ones_like(h2h_src_id) * torch.LongTensor([self.rels.index('h2h')])
            h2h_edge_norm = torch.ones_like(h2h_src_id) * 1.0
            src_id = torch.cat([src_id, h2h_src_id], dim=0)
            dst_id = torch.cat([dst_id, h2h_dst_id], dim=0)
            edge_types = torch.cat([edge_types, h2h_edge_types], dim=0)
            edge_norm = torch.cat([edge_norm, h2h_edge_norm], dim=0)

        edge_norm = edge_norm.unsqueeze(dim=1).float()
        edge_types = edge_types.float()

        self.graph = dgl.graph((src_id, dst_id), num_nodes=total_node_num, idtype=torch.int64)
        self.graph.ndata['h'] = features
        self.graph.edata.update({'rel_type': edge_types, 'norm': edge_norm})
