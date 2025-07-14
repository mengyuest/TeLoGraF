import os
import sys
import random
import argparse
import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import math
import torch
import pytorch_kinematics as pk
import utils
from typing import List
from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle
from generate_scene_v1 import SimpleAnd, SimpleOr, SimpleNot, SimpleListAnd, SimpleListOr, SimpleF, SimpleG, SimpleReach, SimpleUntil,\
    find_ap_in_lines, convert_stl_to_string

import psutil
import gc

def panda_postproc(CA, trajs):
    seg_all_points = CA.endpoint(trajs, all_points=True)
    stl_x = {
        "ee": seg_all_points[-1, :],
        "points": seg_all_points,
        "joints": trajs,
    }
    return stl_x


class ChainAPI():
    def __init__(self, chain, dof, base_pos=None, base_quat=None, device=None):
        self.chain = chain
        self.dof = dof
        # base setup
        self.base_pos = base_pos
        self.base_quat = base_quat
        self.device=device
        if base_pos is not None:
            self.base_mat = self.get_base_mat(base_pos, base_quat)
    
    def get_base_mat(self, base_pos, base_quat):
        list_ = p.getMatrixFromQuaternion(base_quat.cpu().numpy())
        base_rotation_matrix = torch.tensor(list_).float().reshape(3, 3).to(self.device)
        base_translation_matrix = torch.eye(4).to(self.device)
        base_translation_matrix[:3, :3] = base_rotation_matrix
        base_translation_matrix[:3, 3] = base_pos
        return base_translation_matrix
    
    def endpoint(self, s, all_points=False, reverse_cat=False, with_orientation=False):
        shape = list(s.shape)
        s_2d = s.reshape(-1, s.shape[-1])
        # assert s_2d.shape[-1]==self.dof
        M = self.chain.forward_kinematics(s_2d[..., :self.dof], end_only=not all_points)
        if all_points:
            if reverse_cat:
                assert with_orientation==False
                res = torch.stack([M[mk].get_matrix()[..., :3, 3] for mk in list(M)[::-1]], dim=-2)
                return res.reshape(shape[:-1] + [len(M)*3,])
            else:
                if self.base_pos is not None:
                    res_mat = torch.stack([M[mk].get_matrix() for mk in M], dim=0)
                    final_pose = torch.matmul(self.base_mat, res_mat)
                    final_position = final_pose[..., :3, 3].reshape([len(M),] + shape[:-1] + [3,])
                    if with_orientation:
                        final_orient = final_pose[..., :3, :3].reshape([len(M),] + shape[:-1] + [3, 3])
                        return final_position, final_orient
                    else:
                        return final_position
                else:
                    assert with_orientation==False
                    res = torch.stack([M[mk].get_matrix()[..., :3, 3] for mk in M], dim=0)
                    return res.reshape([len(M),] + shape[:-1] + [3,] )
        else:  # end_only=True
            if self.base_pos is not None:
                ee_pose_matrix = M.get_matrix()
                final_pose = torch.matmul(self.base_mat, ee_pose_matrix)
                final_position = final_pose[..., :3, 3].reshape(shape[:-1] + [3,])
                if with_orientation:
                    final_orient = final_pose[..., :3, :3].reshape(shape[:-1] + [3, 3])
                    return final_position, final_orient
                else:
                    return final_position
            else:
                assert with_orientation==False
                res = M.get_matrix()[..., :3, 3]
                return res.reshape(shape[:-1] + [3,] )
    
def get_trajectories(init_x, us, dof, dt):
    segs = [init_x]
    x = segs[-1]
    for ti in range(us.shape[-2]):
        new_x = torch.zeros_like(x)
        new_x[:, :dof] = x[:, :dof] + us[:, ti, :dof] * dt
        segs.append(new_x)
        x = new_x
    return torch.stack(segs, dim=1)

def add_object(config, xyz, euler, color, penetrate=False):
    if penetrate:
        obs_id_coll = -1
    if config[0] == p.GEOM_BOX:
        if not penetrate:
            obs_id_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=config[1:4])
        obs_id_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=config[1:4], rgbaColor=color)
    elif config[0] == p.GEOM_CYLINDER:
        if not penetrate:
            obs_id_coll = p.createCollisionShape(p.GEOM_CYLINDER, height=config[1], radius=config[2])
        obs_id_vis = p.createVisualShape(p.GEOM_CYLINDER, length=config[1], radius=config[2], rgbaColor=color)
    elif config[0] == p.GEOM_SPHERE:
        if not penetrate:
            obs_id_coll = p.createCollisionShape(p.GEOM_SPHERE, radius=config[1])
        obs_id_vis = p.createVisualShape(p.GEOM_SPHERE, radius=config[1], rgbaColor=color)
    else:
        raise NotImplementedError
            
    obs_id = p.createMultiBody(baseMass=0,
                baseCollisionShapeIndex=obs_id_coll,
                baseVisualShapeIndex=obs_id_vis,
                basePosition=xyz,
                baseOrientation=p.getQuaternionFromEuler(euler))

    return obs_id

def render_bullet(angle=None):
    # height = 720
    # width = 960
    # return np.random.uniform(0, 1, (height,width,4))
    if angle=="bev":
        camTargetPos = [0,0,0.6]
        camDistance = 5
        roll = 0
        yaw = 90
        pitch = -90
    elif angle=="custom":
        camTargetPos = [-0.5,0,0.4]
        camDistance = 4.5
        roll = 0
        yaw = 90
        pitch = -20
    elif angle=="left_side":
        camTargetPos = [0,0,0.6]
        camDistance = 5
        roll = 0
        yaw = 0
        pitch = 0
    elif angle=="right_side":
        camTargetPos = [0,0,0.6]
        camDistance = 5
        roll = 0
        yaw = 180
        pitch = 0
    else:
        camTargetPos = [-0.5,0,0.4]
        camDistance = 5
        roll = 0
        yaw = 90
        pitch = -10
    
    upAxisIndex = 2
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)

    if angle=="custom":
        height = 1280
        # height = 960
        width = 1280
    else:
        height = 720
        # height = 960
        width = 960
    nearPlane = 0.001
    farPlane = 100
    fov = 30
    aspect = width / height
    projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

    images = p.getCameraImage(width, height, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix)
    images = np.reshape(images[2], (height, width, 4)) * 1. / 255.
    return images


def interpolate_func(x, N=5):
    # first dim, M->N
    K = x.shape[0]
    interp_points = []
    for i in range(K - 1):
        start = x[i]
        end = x[i + 1]
        # Interpolate N points between start and end
        t = torch.linspace(0, 1, N + 2, device=x.device)[1:-1].view([-1] + [1 for _ in list(x.shape[1:])])
        interp = start * (1 - t) + end * t
        interp_points.append(start.unsqueeze(0))  # Add the starting point
        interp_points.append(interp)  # Add the interpolated points
    interp_points.append(x[-1].unsqueeze(0))  # Add the last point
    return torch.cat(interp_points, dim=0)


def randomly_generate_an_object_on_table(table_height, xmin, xmax, ymin, ymax, a_min=0.03, a_max=0.05, margin=0.05):
    half_a = np.random.uniform(a_min, a_max)
    zmin = zmax = table_height + half_a
    x, y, z, half_a = randomly_generate_an_object(xmin, xmax, ymin, ymax, zmin, zmax, half_a, a_min, a_max, margin)
    return [x, y, z, half_a]

def randomly_generate_an_object_above_table(table_height, xmin, xmax, ymin, ymax, a_min=0.03, a_max=0.05, margin=0.05):
    half_a = np.random.uniform(a_min, a_max)
    zmin = table_height + half_a
    zmax = zmin + 0.5
    x, y, z, half_a = randomly_generate_an_object(xmin, xmax, ymin, ymax, zmin, zmax, half_a, a_min, a_max, margin)
    return [x, y, z, half_a]

def randomly_generate_an_object(xmin, xmax, ymin, ymax, zmin, zmax, half_a=None, a_min=0.03, a_max=0.05, margin=0.05):
    if half_a is None:
        half_a = np.random.uniform(a_min, a_max)
    x = np.random.uniform(xmin+margin+half_a, xmax-margin-half_a)
    y = np.random.uniform(ymin+margin+half_a, ymax-margin-half_a)
    z = np.random.uniform(zmin, zmax)
    return [x, y, z, half_a]

def _random_f_interval(small_a=False, second_half=False):
    if small_a:
        if args.nt>64:
            ta = np.random.randint(0, 16)  # TODO (assign ta, tb)
            tb = np.random.randint(args.nt - 16, args.nt)
        elif args.nt>32:
            ta = np.random.randint(0, 8)  # TODO (assign ta, tb)
            tb = np.random.randint(args.nt - 8, args.nt)
        else:
            ta = np.random.randint(0, 4)  # TODO (assign ta, tb)
            tb = np.random.randint(args.nt - 4, args.nt)
    elif second_half:
        ta = np.random.randint(args.nt//2, args.nt-1)  # TODO (assign ta, tb)
        tb = np.random.randint(ta+1, args.nt)
    else:
        ta = np.random.randint(0, args.nt-1)  # TODO (assign ta, tb)
        tb = np.random.randint(ta+1, args.nt)
    return ta, tb
    
def _random_g_interval():
    tc = 0
    td = np.random.randint(3, 10)
    return tc, td

def _default_g_interval():
    return 0, args.nt

def _entire_interval():
    return 0, args.nt

def cal_dist_approx3d(o1, o2): # consider cube as sphere, and compute their dist
    o1=np.array(o1)
    o2=np.array(o2)
    return np.linalg.norm(o1[:3]-o2[:3], ord=2) - (o1[-1] + o2[-1]) * np.sqrt(3)

def cal_base_dist_approx(obj, base_points):
    # obj is list
    # base_points is tensor
    obj = torch.tensor(obj).to(device)
    dist = torch.min(torch.norm(obj[0:3]-base_points, dim=-1)).item()
    return dist - obj[3] * np.sqrt(3)

def _check_violation(obj, curr_objs, base_points):
    obj_collision = any([cal_dist_approx3d(obj, other_obj) < args.obj_min_gap for other_obj in curr_objs])
    base_collision = cal_base_dist_approx(obj, base_points) < args.obj_base_min_gap
    return obj_collision or base_collision

def _wrap_with_random_stay(tmp_stl):
    stay_mode = np.random.random() > 0.5
    if stay_mode:
        ta = tmp_stl.ts
        tb = tmp_stl.te
        tc = 0
        td = np.random.randint(3, 10)
        reach_stl = tmp_stl.children[0]
        tmp_stl = SimpleF(ta, tb, SimpleG(tc, td, reach_stl))
    else:
        tmp_stl = tmp_stl
    return tmp_stl

def _wrap_with_avoids(tmp_stl, tmp_objects: List, other_configs, num_avoids=None):
    r_min = 0.05
    r_max = 0.10
    base_points, table_height, table_xmin, table_xmax, table_ymin, table_ymax = other_configs
    if num_avoids is None:
        # num_avoids = 0
        num_avoids = np.random.choice(7, p=[0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1])  # 0~6 obstacles
        # num_avoids = np.random.choice(13)  # 0~12 obstacles
    n_tries = 0
    avoids_stls = []
    tmp_goals = [xxx for xxx in tmp_objects]
    tmp_obstacles = []
    while n_tries < args.n_max_tries and len(tmp_obstacles) < num_avoids:
        n_tries += 1
        if np.random.rand()<0.5:
            object = randomly_generate_an_object_above_table(table_height,table_xmin, table_xmax, table_ymin, table_ymax, a_min=r_min, a_max=r_max)  
        else:
            object = randomly_generate_an_object_on_table(table_height,table_xmin, table_xmax, table_ymin, table_ymax, a_min=r_min, a_max=r_max)  
        violation = _check_violation(object, tmp_objects, base_points)
        if violation == False:
            tc, td = _entire_interval()
            sub_stl = SimpleG(tc, td, SimpleNot(SimpleReach(len(tmp_objects), object=object, mode="panda", ap_type=1)))
            avoids_stls.append(sub_stl)
            tmp_obstacles.append(object)
            tmp_objects.append(object)

    if isinstance(tmp_stl, SimpleAnd) or isinstance(tmp_stl, SimpleListAnd):
        tmp_stl = SimpleListAnd(tmp_stl.children + avoids_stls)
    elif len(tmp_obstacles) > 1:
        tmp_stl = SimpleListAnd([tmp_stl] + avoids_stls)
    elif len(tmp_obstacles) == 1:
        tmp_stl = SimpleAnd(tmp_stl, avoids_stls[0])
    else:
        tmp_stl = tmp_stl
    return tmp_stl, tmp_objects, list(range(len(tmp_goals))), [len(tmp_goals)+old_id for old_id in range(len(tmp_obstacles))]


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def collect_new_stl(stl_type_i, trial_i, random_seed, margin, init_x, other_configs):
    ###############################
    # randomly gen object and stl #
    curr_objects = []
    base_points, table_height, table_xmin, table_xmax, table_ymin, table_ymax = other_configs
    if stl_type_i==0: # reach / stay
        n_tries = 0
        while n_tries < args.n_max_tries:
            n_tries += 1
            object = randomly_generate_an_object_on_table(table_height, table_xmin+margin, table_xmax-margin, table_ymin+margin, table_ymax-margin)
            violation = _check_violation(object, curr_objects, base_points)
            if violation == False:
                ta, tb = _random_f_interval(second_half=True)
                final_stl = SimpleF(ta, tb, SimpleReach(len(curr_objects), object=object, mode="panda", ap_type=0))
                curr_objects.append(object)
                final_stl = _wrap_with_random_stay(final_stl)
                final_stl, curr_objects, goals_indices, obstacles_indices = _wrap_with_avoids(final_stl, curr_objects, other_configs)
                break
    elif stl_type_i==1:  # and/or nested
        init_type = np.random.choice(["and", "or"], p=[0.5, 0.5])
        final_stl = SimpleListAnd([]) if init_type=="and" else SimpleListOr([])
        stack = [(0, init_type, final_stl)]
        failure = False
        while len(stack)>0 and failure==False:
            depth, node_type, curr_stl = stack[-1]
            del stack[-1]
            if depth==0:
                pattern_str_list = ["reach,reach", "reach,reach,reach", "_,reach","_,_"]
            elif depth==1:
                if node_type=="and":
                    pattern_str_list = ["reach,reach", "reach,reach"]
                else:
                    pattern_str_list = ["reach,reach", "_,reach", "_,_"]
            else:
                assert depth==2
                pattern_str_list = ["reach,reach"]
            patterns = np.random.choice(pattern_str_list).split(",")
            for ii in range(len(patterns)):
                if patterns[ii]=="_":
                    next_node_type = "or" if node_type=="and" else "and"
                else:
                    next_node_type = "reach"
                if next_node_type=="reach":
                    n_tries = 0
                    while n_tries < args.n_max_tries:
                        n_tries += 1
                        if node_type=="or":
                            object = randomly_generate_an_object_on_table(table_height, table_xmin, table_xmax, table_ymin, table_ymax)
                        else:
                            object = randomly_generate_an_object_on_table(table_height, table_xmin+margin, table_xmax-margin, table_ymin+margin, table_ymax-margin)
                        violation = _check_violation(object, curr_objects, base_points)
                        if violation == False:
                            ta, tb = _random_f_interval()
                            reach_stl = SimpleF(ta, tb, SimpleReach(len(curr_objects), object=object, mode="panda", ap_type=0))
                            next_stl = _wrap_with_random_stay(reach_stl)
                            curr_objects.append(object)
                            break
                    else:
                        failure = violation
                else:
                    if next_node_type == "and":
                        next_stl = SimpleListAnd([])
                    else:
                        next_stl = SimpleListOr([])
                    stack.append([depth+1, next_node_type, next_stl])
                if failure:
                    break
                curr_stl.children.append(next_stl)
        assert failure==False
        final_stl, curr_objects, goals_indices, obstacles_indices = _wrap_with_avoids(final_stl, curr_objects, other_configs)
    elif stl_type_i==2:
        num_eles = np.random.randint(2, 4)  # choose 2~4
        prev_stl = None
        n_tries = 0
        while n_tries < args.n_max_tries and len(curr_objects) < num_eles:
            n_tries += 1
            object = randomly_generate_an_object_on_table(table_height, table_xmin+margin, table_xmax-margin, table_ymin+margin, table_ymax-margin)
            violation = _check_violation(object, curr_objects, base_points)
            if violation == False:
                ta, tb = _random_f_interval(small_a=True)
                sub_stl = SimpleF(ta, tb, SimpleReach(len(curr_objects), object=object, mode="panda", ap_type=0))
                sub_stl = _wrap_with_random_stay(sub_stl)
                if prev_stl is not None:
                    and_stl = SimpleAnd(left=sub_stl.children[0], right=prev_stl)
                    sub_stl.children = [and_stl]
                prev_stl = sub_stl
                curr_objects.append(object)
        final_stl = sub_stl
        final_stl, curr_objects, goals_indices, obstacles_indices = _wrap_with_avoids(final_stl, curr_objects, other_configs)
    elif stl_type_i==3:
        until_patterns = [
            # 2
            ["A,B", "A"],
            # 3
            ["A,B", "B,C", "A"],
            ["A,B", "A,C", "A"],
            ["B,A", "C,A", "B", "C"],
            # 4
            ["A,B", "C,D", "A", "C"],
        ]
        until_pattern = until_patterns[np.random.choice(len(until_patterns))]
        exist_objs = {}
        until_stls = []
        reach_stls = []
        
        for ele in until_pattern:
            if "," in ele:
                ele1, ele2 = ele.split(",")
                for _ele in [ele1, ele2]:
                    if _ele not in exist_objs:
                        n_tries = 0
                        while n_tries < args.n_max_tries:
                            n_tries += 1
                            object = randomly_generate_an_object_on_table(table_height, table_xmin+margin, table_xmax-margin, table_ymin+margin, table_ymax-margin)
                            violation = _check_violation(object, curr_objects, base_points)
                            if violation == False:
                                curr_objects.append(object)
                                break
                        assert violation==False
                        exist_objs[_ele] = len(curr_objects)-1
                tc, td = _default_g_interval()
                id0 = exist_objs[ele1]
                id1 = exist_objs[ele2]
                until_stl = SimpleUntil(tc, td, 
                                        SimpleNot(SimpleReach(id0, object=curr_objects[id0], mode="panda", ap_type=0)),
                                        SimpleReach(id1, object=curr_objects[id1], mode="panda", ap_type=0)
                                        )
                until_stls.append(until_stl)
            else:
                id0 = exist_objs[ele]
                ta, tb = _entire_interval() # TODO here we consider a loose case
                reach_stl = SimpleF(ta, tb, SimpleReach(id0, object=curr_objects[id0], mode="panda", ap_type=0))
                reach_stl = _wrap_with_random_stay(reach_stl)
                reach_stls.append(reach_stl)
            
        final_stl = SimpleListAnd(until_stls + reach_stls)
        final_stl, curr_objects, goals_indices, obstacles_indices = _wrap_with_avoids(final_stl, curr_objects, other_configs)
    else:
        raise NotImplementedError
    
    stl_np_lines = convert_stl_to_string(final_stl, numpy=True)
    record = {
        "stl_type_i": stl_type_i,
        "trial_i": trial_i,
        "seed": random_seed,
        "ego": utils.to_np(init_x),
        "stl": stl_np_lines, 
        "objects": curr_objects, 
        "goals_indices": goals_indices, 
        "obstacles_indices": obstacles_indices,
    }
    
    return record, final_stl


def _mean(xlist):
    if len(xlist)==0:
        return 0
    else:
        return np.mean(xlist)


def main():   
    utils.setup_exp_and_logger(args, test=args.test)
    VIZ_PATH = args.viz_dir
    
    NUM_CASES = 4
    # setup the robot environment
    p.connect(p.DIRECT)
    
    # Base pose (position and orientation)
    base_position = torch.tensor([0, 0, 0.6])  # [x, y, z]
    base_position_np = base_position.cpu().numpy()
    base_orientation_quat = torch.tensor(p.getQuaternionFromEuler([0, 0, 0]))  # Quaternion (w, x, y, z)
    base_orientation_quat_np = base_orientation_quat.cpu().numpy()
    
    # robot setup
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # TABLE CONFIGS
    # # each cell 0.9m*0.9m
    # # table range (0<x<0.9, -0.75<y<0.75, height=0.62)
    table_height = 0.62
    table_xmin = 0
    table_xmax = 0.90
    table_ymin = -0.75
    table_ymax = 0.75
    
    dof = 7
    thmin = np.array([-166,-101,-166,-176,-166,-1,-166]) / 180 * np.pi
    thmax = np.array([166,101,166,-4,166,215,166]) / 180 * np.pi
    
    search_path = pybullet_data.getDataPath()
    chain = pk.build_serial_chain_from_urdf(
        open(os.path.join(search_path, "franka_panda/panda.urdf")).read(), "panda_link8")
    chain = chain.to(device=device)
    CA = ChainAPI(chain, dof, base_position, base_orientation_quat, device=device)
    
    thmin_torch = torch.tensor([-166,-101,-166,-176,-166,-1,-166]).float() / 180 * np.pi
    thmax_torch = torch.tensor([166,101,166,-4,166,215,166]).float() / 180 * np.pi
    
    n_table_pts = 500
    xmin = table_xmin
    xmax = table_xmax
    ymin = table_ymin
    ymax = table_ymax
    zmin = table_height - 0.02
    zmax = table_height + 0.02
    table_point_cloud = torch.cat([
        utils.uniform_tensor(xmin, xmax, (n_table_pts, 1)),
        utils.uniform_tensor(ymin, ymax, (n_table_pts, 1)),
        utils.uniform_tensor(zmin, zmax, (n_table_pts, 1)),
    ], dim=-1)
    table_point_cloud = utils.to_np(table_point_cloud)
    
    margin = 0.2
    process = psutil.Process()
    
    eta = utils.EtaEstimator(0, end_iter=(args.num_trials-args.start_from)*NUM_CASES)
    exp_idx = 0
    data_list = []
    
    init_x_original = ((thmin_torch + thmax_torch) / 2).to(device)
    init_x_original[3] = -2.0
    # init_x_original[5] = 2.2
    
    base_points = interpolate_func(CA.endpoint(init_x_original, all_points=True), N=5)
    other_configs = base_points, table_height, table_xmin, table_xmax, table_ymin, table_ymax
    
    acc_list={
        "all":[],
        0:[], 1:[], 2:[], 3:[],
    }
    
    acc_case_list={
        "all":[],
        0:[], 1:[], 2:[], 3:[],
    }
    
    for trial_i in range(args.start_from, args.num_trials):
        for stl_type_i in args.cases: # [0, 1, 2, 3]:
            eta.update()
            if args.base_seed is not None:
                seedseed = args.base_seed + trial_i * 4 + stl_type_i
            else:
                seedseed = args.seed * args.num_trials * 4 + trial_i * 4 + stl_type_i
            utils.seed_everything(seedseed)
            
            ################
            # generate stl #
            record, final_stl = \
                collect_new_stl(stl_type_i, trial_i, seedseed, margin, init_x_original, other_configs)
            data_list.append(record)
            curr_objects = record["objects"]
            stl_np_lines = record["stl"]
            goals_indices = record["goals_indices"]
            obstacles_indices = record["obstacles_indices"]
            
            stl_dict = {}
            objects_d={}
            real_stl = find_ap_in_lines(0, stl_dict=stl_dict, objects_d=objects_d, lines=stl_np_lines, numpy=True, real_stl=True, ap_mode="panda", until1=False)
            avoid_loc_list = torch.tensor([curr_objects[coll_idx] for coll_idx in obstacles_indices]).to(device)
            
            n_uinits = 64
            init_x = init_x_original[None, :].repeat(n_uinits, 1).to(device)
            us_raw = torch.randn(n_uinits, args.nt, dof).float().to(device).requires_grad_(True)
            optimizer = torch.optim.Adam([us_raw], lr=args.trajopt_lr)
            for iter_i in range(args.trajopt_niters):
                factored_us = torch.tanh(us_raw * 1) * args.u_max
                
                trajs = get_trajectories(init_x, factored_us, dof, args.dt)
                
                seg_all_points = CA.endpoint(trajs, all_points=True)
                
                stl_x = {
                    "ee": seg_all_points[-1, :],
                    "points": seg_all_points,
                    "joints": trajs,
                }
                
                stl_scores = real_stl(stl_x, tau=500)[:, 0]
                acc_avg = torch.mean((stl_scores>0).float())
                stl_loss = torch.mean(torch.nn.ReLU()(0.5 - stl_scores))
                
                dense_point = interpolate_func(seg_all_points, N=4)
                if len(obstacles_indices)>0:
                    avoid_obj_r = avoid_loc_list[:, 3]
                    safe_r = 0.0
                    avoid_loss = torch.sum(torch.nn.ReLU()(avoid_obj_r + safe_r - torch.norm(dense_point[..., None, :]-avoid_loc_list[:, 0:3],dim=-1)))
                else:
                    avoid_loss = 0 * stl_loss
                reg_loss = 0
                for i in range(dof):
                    reg_loss_item1 = torch.nn.ReLU()(thmin[i] - trajs[..., i])
                    reg_loss_item2 = torch.nn.ReLU()(trajs[..., i] - thmax[i])
                    reg_loss = torch.mean(reg_loss + reg_loss_item1 + reg_loss_item2)
                above_mid_air_loss = torch.sum(torch.nn.ReLU()(0.8 - seg_all_points[[4,5,6], ..., 2])) * 1
                above_table_loss = torch.sum(torch.nn.ReLU()(0.65 - seg_all_points[2:, ..., 2])) * 10
                # (M, B, T+1, 3)
                smooth_loss = torch.mean(torch.square(factored_us[:, 1:]-factored_us[:, :-1])) * 0.2
                loss = stl_loss + reg_loss + above_mid_air_loss + above_table_loss + avoid_loss + smooth_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if iter_i % 10 == 0 and args.quiet==False:
                    print("Trial-%05d type:%d Iter-%05d Loss:%.4f  stl:%.4f acc:%.3f reg:%.4f aboe:%.4f air:%.4f avoid:%.4f smooth:%.4f | dT:%s  Elapsed:%s  ETA:%s"%(
                        trial_i, stl_type_i, iter_i, loss.item(), stl_loss.item(), acc_avg.item(), reg_loss.item(), 
                        above_table_loss.item(), above_mid_air_loss.item(), avoid_loss.item(),
                        smooth_loss.item(), eta.interval_str(), eta.elapsed_str(), eta.eta_str(),
                        ))
            
            ###############
            # save record #
            record["score"] = utils.to_np(stl_scores)
            record["us"] = utils.to_np(factored_us)
            record["state"] = utils.to_np(trajs[..., 0, :])
            record["trajs"] = utils.to_np(trajs)
            
            case_avg_acc = acc_avg.item()
            case_acc = int(acc_avg.item()>0)
            acc_list["all"].append(case_avg_acc)
            acc_list[stl_type_i].append(case_avg_acc)
            acc_case_list["all"].append(case_acc)
            acc_case_list[stl_type_i].append(case_acc)
            if trial_i % 10 == 0 or trial_i==args.num_trials-1:
                print("Trial-%04d  seed:%d | ACC:%.4f (%.4f %.4f %.4f %.4f) CASE: %.4f (%.4f %.4f %.4f %.4f) | dT:%s  Elapsed:%s  ETA:%s"%(
                    trial_i, seedseed,
                    _mean(acc_list["all"]), _mean(acc_list[0]), _mean(acc_list[1]), _mean(acc_list[2]), _mean(acc_list[3]), 
                    _mean(acc_case_list["all"]), _mean(acc_case_list[0]), _mean(acc_case_list[1]), _mean(acc_case_list[2]), _mean(acc_case_list[3]),
                    eta.interval_str(), eta.elapsed_str(), eta.eta_str(),
                ))
            
            if trial_i<args.n_viz:
                p.resetSimulation()
                robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION, 
                          basePosition=base_position_np, baseOrientation=base_orientation_quat_np)
    
                # background setup
                plane_id = p.loadURDF("plane.urdf")
                table_id = p.loadURDF("table/table.urdf", basePosition=[0.4, 0, 0], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))
                
                #################
                # visualization #
                sidx = torch.where(stl_scores>0)[0]
                if len(sidx)==0:
                    sidx = 0
                else:
                    sidx = sidx[0]
                
                xyz_traj_np = utils.to_np(seg_all_points)
                fig = plt.figure(figsize=(8, 8))
                for joint_i in range(xyz_traj_np.shape[0]):
                    plt.plot(range(args.nt+1), xyz_traj_np[joint_i, sidx, :, 2]+0.02*np.random.rand(), label=str(joint_i))
                plt.axhline(y=0.8, linestyle="--", color="gray")
                plt.axhline(y=0.65, linestyle="--", color="brown")
                plt.legend()
                plt.savefig("%s/viz_horizon_e%04d_%d.png"%(VIZ_PATH, trial_i, stl_type_i), bbox_inches='tight', pad_inches=0.1)
                plt.clf()
                fig.clear()
                plt.close()
                gc.collect()
                
                ##############################
                # update pybullet simulation #
                obj_id_list = []
                for obj_i, obj in enumerate(curr_objects):
                    if obj_i not in obstacles_indices:
                        obj_id = add_object([p.GEOM_BOX, obj[-1], obj[-1], obj[-1]], [obj[0], obj[1], obj[2]], [0,0,0], [0.5,0,1,0.3], penetrate=False)
                    else:
                        obj_id = add_object([p.GEOM_SPHERE, obj[-1]], [obj[0], obj[1], obj[2]], [0,0,0], [1,0,0,0.8], penetrate=False)
                    obj_id_list.append(obj_id)
                
                for ti in range(args.viz_nt):
                    if ti%1==0 or ti==args.nt-1:
                        for dof_i in range(dof):
                            p.resetJointState(robot_id, dof_i, targetValue=trajs[sidx, ti, dof_i].item())

                        fig = plt.figure(figsize=(8, 6))
                        # images = render_bullet()
                        # plt.imshow(images)
                        plt.subplot(2, 3, 1)
                        images = render_bullet()
                        plt.imshow(images)
                        plt.axis("off")
                        plt.subplot(2, 3, 2)
                        images = render_bullet(angle="bev")
                        plt.imshow(images)
                        plt.axis("off")
                        plt.subplot(2, 3, 3)
                        images = render_bullet(angle="left_side")
                        plt.imshow(images)
                        plt.axis("off")
                        plt.subplot(2, 3, 5)
                        images = render_bullet(angle="right_side")
                        plt.imshow(images)
                        plt.axis("off")
                        ax = fig.add_subplot(2, 3, 4, projection='3d')
                        seg_all_points_np = utils.to_np(seg_all_points)
                        ax.scatter(table_point_cloud[:, 0], table_point_cloud[:, 1], table_point_cloud[:, 2], color="brown", alpha=0.8, s=5, zorder=999999)
                        ax.set_box_aspect([1,1,1])
                        ax.set_xlim(table_xmin-0.1, table_xmax+0.1)
                        ax.set_ylim(table_ymin-0.1, table_ymax+0.1)
                        ax.set_zlim(table_height-0.2, table_height+1.0)
                        
                        
                        viz_points = seg_all_points_np[:, sidx, ti, :]
                        
                        ax.plot3D(viz_points[:, 0], viz_points[:, 1], viz_points[:, 2], color="gray")
                        ax.scatter(viz_points[-1, 0], viz_points[-1, 1], viz_points[-1, 2], color="magenta")
                        
                        plt.subplot(2, 3, 6)
                        ax = plt.gca()
                        for obj_i, obj in enumerate(curr_objects):
                            x, y, z, r =obj
                            if obj_i in goals_indices:
                                rect = Rectangle([y-r, x-r], 2*r, 2*r, color="green", alpha=0.4, zorder=99999)
                                ax.add_patch(rect)
                            else:
                                circ = Circle([y, x], r, color="red", alpha=0.4, zorder=99999)
                                ax.add_patch(circ)
                            plt.text(y, x, str(obj_i))
                        
                        n_traj_viz=8
                        des_stl_indices=torch.argsort(stl_scores, descending=True)
                        sidx = torch.argsort(stl_scores)[0]
                        ee_np = utils.to_np(stl_x["ee"])
                        for viz_traj_i in range(n_traj_viz):
                            viz_idx = des_stl_indices[viz_traj_i]
                            plt.plot(ee_np[viz_idx, :, 1], ee_np[viz_idx, :, 0], color="royalblue", alpha=0.6, linewidth=0.5)
                        
                        plt.axis("scaled")
                        plt.ylim(table_xmin, table_xmax)
                        plt.xlim(table_ymin, table_ymax)
                        plt.tight_layout()
                        plt.savefig("%s/viz_e%04d_%d_%04d.png"%(VIZ_PATH, trial_i, stl_type_i, ti), bbox_inches='tight', pad_inches=0.1)
                        
                        # TODO (not sure why but this solves memory leak)
                        plt.clf()
                        fig.clear()
                        plt.close()
                        gc.collect()
                
                if args.gen_gif and args.viz_nt>0:
                    os.system("convert -delay 5 -loop 0 %s/viz_e%04d_%d*.png %s/panda_e%04d_%d.gif"%(
                        VIZ_PATH, trial_i, stl_type_i,
                        VIZ_PATH, trial_i, stl_type_i))

            if args.print_mem:
                print("*"*50)
                print("AFTER",process.memory_info().rss / 1024 ** 3, "GB")  # in bytes 
            
            # save data
            if exp_idx % args.save_freq == 0 or exp_idx == args.num_trials * NUM_CASES - 1:
                np.savez("%s/data.npz"%(args.exp_dir_full), data=data_list)
            if trial_i==100:
                np.savez("%s/data100.npz"%(args.exp_dir_full), data=data_list)    
            exp_idx += 1
          
    np.savez("%s/data.npz"%(args.exp_dir_full), data=data_list)     
    p.disconnect()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--seed", type=int, default=1007)
    add("--exp_name", '-e', type=str, default="data_panda_DBG")
    add("--gpus", type=str, default="0")
    add("--cpu", action='store_true', default=False)
    add("--test", action='store_true', default=False)
    add("--dryrun", action='store_true', default=False)
    add("--num_trials", type=int, default=100)
    add("--nt", type=int, default=64)
    add("--dt", type=float, default=0.05)
    add("--no_viz", action='store_true', default=False)
    add("--viz_freq", type=int, default=50)
    add("--print_freq", type=int, default=10)
    add("--save_freq", type=int, default=10)
    add("--start_from", type=int, default=0)
    add("--n_max_tries", type=int, default=1000)
    add("--trajopt_lr", type=float, default=3e-2)
    add("--trajopt_niters", type=int, default=500)
    add("--obj_min_gap", type=float, default=0.01)
    add("--obj_base_min_gap", type=float, default=0.3)
    
    add("--cases", type=int, nargs="+", default=[0,1,2,3])
    add("--print_mem", action='store_true', default=False)
    
    add("--u_max", type=float, default=1.0)
    
    add("--n_viz", type=int, default=3)
    add("--viz_nt", type=int, default=None)
    add("--gen_gif", action='store_true', default=False)
    add("--quiet", action='store_true', default=False)
    
    add("--base_seed", type=int, default=None)
    
    args = parser.parse_args()
    
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda:0"
    if args.viz_nt is None:
        args.viz_nt = args.nt
    print("python", " ".join(sys.argv))
    t1 = time.time()
    main()
    t2 = time.time()
    print("Finished in %.3f seconds"%(t2-t1))