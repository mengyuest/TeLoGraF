import os
import torch
import argparse
import time
import numpy as np
import utils
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle
from stl_d_lib import * 
import textwrap
from typing import List
# STL store format
# node_id, node_type, time_window, len(children), [children_id_list]
# node_type - semantics
# 0 - AND
# 1 - OR 
# 2 - NOT 
# 3 - Imply
# 4 - Next
# 5 - Eventually
# 6 - Always
# 7 - Until
# 8 - AP's / objects

# TODO (single/double-integrator, dubins-car)
def dynamics(s, u, dt):
    new_s = torch.stack([
        s[..., 0] + u[..., 0] * dt, 
        s[..., 1] + u[..., 1] * dt], dim=-1)
    return new_s

def generate_trajectories(s, us, dt, v_max=None):
    new_ds = torch.cumsum(us, dim=-2) * dt
    trajs = s[..., None, :] + new_ds
    trajs = torch.cat([s[..., None, :], trajs], dim=-2)
    return trajs

def generate_trajectories_dubins(s, us, dt, v_max, unclip=False):
    trajs=[s]
    for ti in range(us.shape[-2]):
        prev_s = trajs[-1]
        new_x = prev_s[..., 0] + prev_s[..., 3] * torch.cos(prev_s[..., 2]) * dt
        new_y = prev_s[..., 1] + prev_s[..., 3] * torch.sin(prev_s[..., 2]) * dt
        new_th = (prev_s[..., 2] + us[..., ti, 0] * dt + np.pi) % (2 * np.pi) - np.pi
        if unclip:
            new_v = prev_s[..., 3] + us[..., ti, 1] * dt
        else:
            new_v = torch.clip(prev_s[..., 3] + us[..., ti, 1] * dt, -0.01, v_max)
        new_s = torch.stack([new_x, new_y, new_th, new_v], dim=-1)
        trajs.append(new_s)
    trajs = torch.stack(trajs, dim=-2)
    return trajs

def check_stl_type(node):
    if isinstance(node, SimpleAnd) or isinstance(node, SimpleListAnd):
        node_type=0
    elif isinstance(node, SimpleOr) or isinstance(node, SimpleListOr):
        node_type=1
    elif isinstance(node, SimpleNot):
        node_type=2
    elif isinstance(node, SimpleImply):
        node_type=3
    elif isinstance(node, SimpleNext):
        node_type=4
    elif isinstance(node, SimpleF):
        node_type=5
    elif isinstance(node, SimpleG):
        node_type=6
    elif isinstance(node, SimpleUntil):
        node_type=7
    elif isinstance(node, SimpleReach):
        node_type=8
    else:
        raise NotImplementedError
    return node_type

def convert_stl_to_string(stl, numpy=False):
    id=0
    last_id=0
    queue = [(stl, id)]
    lines=[]
    while len(queue)>0:
        node, id = queue[0]
        del queue[0]
        node_type = check_stl_type(node)
        if numpy:
            curr_s = [id, node_type, node.ts, node.te, len(node.children)]
            if node_type!=8:
                append_s = [last_id+new_i+1 for new_i in range(len(node.children))]
            else:
                if node.ap_type is not None:
                    append_s = [node.obj_id, node.obj_x, node.obj_y, node.obj_z, node.obj_r, node.ap_type]
                else:
                    append_s = [node.obj_id, node.obj_x, node.obj_y, node.obj_z, node.obj_r]
            newline = curr_s + append_s
        else:
            curr_s = "%d %d %d %d %d" % (id, node_type, node.ts, node.te, len(node.children))
            if node_type!=8:
                append_s = " ".join(["%d"%(last_id+new_i+1) for new_i in range(len(node.children))])
            else:
                if node.ap_type is not None:
                    append_s = "%d %.4f %.4f %.4f %.4f %d"%(node.obj_id, node.obj_x, node.obj_y, node.obj_z, node.obj_r, node.ap_type)
                else:
                    append_s = "%d %.4f %.4f %.4f %.4f"%(node.obj_id, node.obj_x, node.obj_y, node.obj_z, node.obj_r)
            newline = curr_s + " " + append_s
        lines.append(newline)
        if node_type!=8:
            for new_i, child in enumerate(node.children):
                queue.append((child, last_id+new_i+1))
            last_id = last_id + len(node.children)
    return lines

def save_stl_to_file(stl, filepath, ego_xy=None, lines=None):
    if lines is None:
        lines = convert_stl_to_string(stl)
    with open(filepath, "w") as f:
        if ego_xy is not None:
            f.write("#ego:%.4f %.4f\n"%(ego_xy[0], ego_xy[1]))
        for line in lines:
            f.write(line+"\n")

def load_stl_from_file(filepath, ap_mode="l2", until1=False):
    stl_dict = {}
    objects_d={}
    lines = open(filepath).readlines()
    if lines[0].startswith("#"):
        ego_xy = np.array([float(xx) for xx in lines[0].strip().split(":")[1].split()])
    else:
        ego_xy = None
    stl = find_ap_in_lines(0, stl_dict, objects_d, lines, ap_mode=ap_mode, until1=until1)
    return stl, ego_xy

def find_ap_in_lines(id, stl_dict, objects_d, lines, numpy=False, real_stl=False, ap_mode="l2", until1=False):
    for line in lines:
        if numpy:
            res = line
        else:
            res = line.strip().split()
        node_id = int(res[0])
        if node_id==id:
            node_type = int(res[1])
            
            # non-object formulas
            if node_type!=8:
                node_ts = int(res[2])
                node_te = int(res[3])
                len_children = int(res[4])
                children_ids = [int(res_id_s) for res_id_s in res[5:]]
                children = []
                for child_id in children_ids:
                    if child_id not in stl_dict:
                        child = find_ap_in_lines(child_id, stl_dict, objects_d, lines, numpy=numpy, real_stl=real_stl, ap_mode=ap_mode, until1=until1)
                        if isinstance(child, SimpleReach): # TODO use node_type as NOT to detect this is an obstacle
                            objects_d[child.obj_id] = {"x":child.obj_x, "y":child.obj_y, "z":child.obj_z, "r":child.obj_r, "is_obstacle":int(node_type)==2}
                        stl_dict[child_id] = child
                    children.append(stl_dict[child_id])
            if real_stl:
                if node_type==0:
                    if len_children==2:
                        stl = And(lhs=children[0], rhs=children[1])
                    else:
                        stl = ListAnd(lists=children)
                elif node_type==1:
                    if len_children==2:
                        stl = Or(lhs=children[0], rhs=children[1])
                    else:
                        stl = ListOr(lists=children)
                elif node_type==2:
                    stl = Not(node=children[0])
                elif node_type==3:
                    stl = Imply(lhs=children[0], rhs=children[1])
                elif node_type==4:
                    stl = Eventually(1, 2, node=children[0])
                elif node_type==5:
                    stl = Eventually(ts=node_ts, te=node_te, node=children[0])
                elif node_type==6:
                    stl = Always(ts=node_ts, te=node_te, node=children[0])
                elif node_type==7:
                    if until1:
                        stl = Until1(ts=node_ts, te=node_te, lhs=children[0], rhs=children[1])
                    else:
                        stl = Until(ts=node_ts, te=node_te, lhs=children[0], rhs=children[1])
                elif node_type==8:
                    obj_id = int(res[5])
                    obj_x = float(res[6])
                    obj_y = float(res[7])
                    obj_z = float(res[8])
                    obj_r = float(res[9])
                    if len(res)==11:
                        ap_type = int(res[10])
                    if ap_mode=="l2":
                        stl = AP(lambda x:obj_r**2 - (x[...,0]-obj_x)**2 - (x[..., 1]-obj_y)**2, comment="R%d"%(obj_id))
                    elif ap_mode=="grid":
                        stl = AP(lambda x:obj_r * 0.5 - torch.maximum(torch.abs(x[...,0]-obj_x), torch.abs(x[..., 1]-obj_y)), comment="R'%d"%(obj_id))
                    elif ap_mode=="panda":
                        if ap_type==0:  # reach an object
                            stl = And(
                                AP(reach_obj_from_panda_decorator(res), comment="R%d"%(obj_id)),
                                AP(reach_obj_from_panda_vert_decorator(res), comment="V%d"%(obj_id))
                            )
                        elif ap_type==1:  # avoid an obstacle's reach
                            stl = AP(reach_obj_from_panda_big_decorator(res), comment="R%d"%(obj_id))
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            else:
                if node_type==0:
                    if len_children==2:
                        stl = SimpleAnd(left=children[0], right=children[1])
                    else:
                        stl = SimpleListAnd(list_aps=children)
                elif node_type==1:
                    if len_children==2:
                        stl = SimpleOr(left=children[0], right=children[1])
                    else:
                        stl = SimpleListOr(list_aps=children)
                elif node_type==2:
                    stl = SimpleNot(ap=children[0])
                elif node_type==3:
                    stl = SimpleImply(left=children[0], right=children[1])
                elif node_type==4:
                    stl = SimpleNext(ap=children[0])
                elif node_type==5:
                    stl = SimpleF(ts=node_ts, te=node_te, ap=children[0])
                elif node_type==6:
                    stl = SimpleG(ts=node_ts, te=node_te, ap=children[0])
                elif node_type==7:
                    stl = SimpleUntil(ts=node_ts, te=node_te, left=children[0], right=children[1])
                elif node_type==8:
                    obj_id = int(res[5])
                    obj_x = float(res[6])
                    obj_y = float(res[7])
                    obj_z = float(res[8])
                    obj_r = float(res[9])
                    stl = SimpleReach(obj_id=obj_id, obj_x=obj_x, obj_y=obj_y, obj_z=obj_z, obj_r=obj_r)
                else:
                    raise NotImplementedError
            break
    stl_dict[id] = stl
    return stl


def get_current_objects_from_stl_lines(lines, numpy=False):
    current_objects = []
    for line in lines:
        if line[1]==8:
            current_objects.append(np.array([line[-4], line[-3], line[-1]]))
    return current_objects

def reach_obj_from_panda_decorator(res):
    obj_id = int(res[5])
    obj_x = float(res[6])
    obj_y = float(res[7])
    obj_z = float(res[8])
    obj_r = float(res[9])
    
    def reach_obj_from_panda(x):
        # x should be a dict
        # x["ee"] end effector (B, T+1, 3)
        # x["points"] joint points (M, B, T+1, 3)
        # x["joints"] joint angles (B, T+1, dof)
        # lambda x:obj_r**2 - (x[...,0]-obj_x)**2 - (x[..., 1]-obj_y)**2
        z_offset = 0.1
        return ((obj_r**2) - (x["ee"][...,0]-obj_x)**2 - (x["ee"][..., 1]-obj_y)**2 - (x["ee"][..., 2]-obj_z-z_offset)**2)/((obj_r))
    return reach_obj_from_panda

def reach_obj_from_panda_big_decorator(res):
    obj_id = int(res[5])
    obj_x = float(res[6])
    obj_y = float(res[7])
    obj_z = float(res[8])
    obj_r = float(res[9])
    
    def reach_obj_from_panda_big(x):
        # x should be a dict
        # x["ee"] end effector (B, T+1, 3)
        # x["points"] joint points (M, B, T+1, 3)
        # x["joints"] joint angles (B, T+1, dof)
        # lambda x:obj_r**2 - (x[...,0]-obj_x)**2 - (x[..., 1]-obj_y)**2
        return (3*(obj_r**2) - (x["ee"][...,0]-obj_x)**2 - (x["ee"][..., 1]-obj_y)**2 - (x["ee"][..., 2]-obj_z)**2)/(np.sqrt(3)*(obj_r))
    return reach_obj_from_panda_big

def reach_obj_from_panda_vert_decorator(res):
    obj_id = int(res[5])
    obj_x = float(res[6])
    obj_y = float(res[7])
    obj_z = float(res[8])
    obj_r = float(res[9])
    
    def reach_obj_from_panda_vert(x):
        # x should be a dict
        # x["ee"] end effector (B, T+1, 3)
        # x["points"] joint points (M, B, T+1, 3)
        # x["joints"] joint angles (B, T+1, dof)
        # lambda x:obj_r**2 - (x[...,0]-obj_x)**2 - (x[..., 1]-obj_y)**2
        vector = x["points"][7, :] - x["points"][8, :]
        vector = vector / torch.clip(torch.norm(vector, dim=-1, keepdim=True), min=1e-4, max=1e4)
        # unit_vector = torch.tensor([[[0, 0, 1.]]]).to(vector.device)
        vertical_val = vector[..., 2] - 0.9
        return vertical_val
    return reach_obj_from_panda_vert

# only takes notes for storage
class SimpleSTL:
    def __init__(self):
        self.children = []
        self.ts = -1
        self.te = -1
        
    def print_out(self, mode="simple"):
        s = self.__class__.__name__
        s_else = ",".join([child.__class__.__name__ for child in self.children])
        print(s + " | " + s_else)
        for child in self.children:
            child.print_out(mode)
        
class SimpleAnd(SimpleSTL):
    def __init__(self, left, right):
        super().__init__()
        self.children.append(left)
        self.children.append(right)

class SimpleListAnd(SimpleSTL):
    def __init__(self, list_aps):
        super().__init__()
        for ap in list_aps:        
            self.children.append(ap)

class SimpleOr(SimpleSTL):
    def __init__(self, left, right):
        super().__init__()
        self.children.append(left)
        self.children.append(right)

class SimpleListOr(SimpleSTL):
    def __init__(self, list_aps):
        super().__init__()
        for ap in list_aps:        
            self.children.append(ap)

class SimpleNot(SimpleSTL):
    def __init__(self, ap):
        super().__init__()
        self.children.append(ap)

class SimpleImply(SimpleSTL):
    def __init__(self, left, right):
        super().__init__()
        self.children.append(left)
        self.children.append(right)

class SimpleNext(SimpleSTL):
    def __init__(self, ap):
        super().__init__()
        self.children.append(ap)

class SimpleF(SimpleSTL):
    def __init__(self, ts, te, ap):
        super().__init__()
        self.ts = ts
        self.te = te
        self.children.append(ap)
    
    def print_out(self, mode="simple"):
        s = self.__class__.__name__
        if len(self.children)==1 and isinstance(self.children[0], SimpleReach):
            if mode=="simple":
                print(s + "(" + "Reach" + str(self.children[0].obj_id) + ")")
            else:
                print("%s[%d,%d] Reach (%s)"%(s, self.ts, self.te, self.children[0].obj_id))
        else:
            s_else = ",".join([child.__class__.__name__ for child in self.children])
            if mode=="simple":
                print(s + " | " + s_else)
            else:
                print("%s[%d,%d] | %s"%(s, self.ts, self.te, s_else))
            for child in self.children:
                child.print_out(mode)

class SimpleG(SimpleSTL):
    def __init__(self, ts, te, ap):
        super().__init__()
        self.ts = ts
        self.te = te
        self.children.append(ap)
        
    def print_out(self, mode="simple"):
        s = self.__class__.__name__
        s_else = ",".join([child.__class__.__name__ for child in self.children])
        if mode=="simple":
            print(s + " | " + s_else)
        else:
            print("%s[%d,%d] | %s"%(s, self.ts, self.te, s_else))
        for child in self.children:
            child.print_out(mode)

class SimpleUntil(SimpleSTL):
    def __init__(self, ts, te, left, right):
        super().__init__()
        self.ts = ts
        self.te = te
        self.children.append(left)
        self.children.append(right)

class SimpleReach(SimpleSTL):
    def __init__(self, obj_id, obj_x=None, obj_y=None, obj_z=None, obj_r=None, object=None, mode="2d", ap_type=None):
        super().__init__()
        self.obj_id = obj_id
        if object is not None:
            self.obj_x = object[0]
            self.obj_y = object[1]
            if mode=="2d":
                self.obj_z = 0
                self.obj_r = object[2]
                self.ap_type = ap_type
            else:
                assert mode=="panda"
                self.ap_type = ap_type
                self.obj_z = object[2]
                self.obj_r = object[3]
        else:
            self.obj_x = obj_x
            self.obj_y = obj_y
            self.obj_z = obj_z
            self.obj_r = obj_r
    
    def print_out(self, mode="simple"):
        s = self.__class__.__name__
        print(s + " obj_id: " + str(self.obj_id))

def cal_dist(o1, o2):
    return np.linalg.norm(o1[:2]-o2[:2], ord=2) - o1[-1] - o2[-1]

def cal_dist2(o1, o2):
    return np.linalg.norm(o1[:2]-o2[:2], ord=2) - o2[-1]

def _random_object(big=False, goals=None):
    if big:
        object_r = np.random.uniform(args.r_min * 0.75 + args.r_max * 0.25, args.r_max)
    else:
        object_r = np.random.uniform(args.r_min, args.r_min * 0.75 + args.r_max * 0.25)
    if goals is not None:
        obj1, obj2 = goals
        ratio = np.random.rand() * 0.3 + 0.35
        dx = (np.random.rand()-0.5) * 0.25
        dy = (np.random.rand()-0.5) * 0.25
        object_x = ratio * obj1[0] + (1-ratio) * obj2[0] + dx
        object_y = ratio * obj1[1] + (1-ratio) * obj2[1] + dy
    else:
        object_x = np.random.uniform(args.x_min + object_r * args.bloat_ratio, args.x_max - object_r * args.bloat_ratio)
        object_y = np.random.uniform(args.y_min + object_r * args.bloat_ratio, args.y_max - object_r * args.bloat_ratio)
    return np.array([object_x, object_y, object_r])


def _random_f_interval(small_a=False):
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
    else:
        ta = np.random.randint(0, args.nt-1)  # TODO (assign ta, tb)
        tb = np.random.randint(ta+1, args.nt)
    return ta, tb
    
def _random_g_interval():
    tc = 0
    td = np.random.randint(3, 10)
    return tc, td

def _default_g_interval():
    return 0, args.nt-1

def _entire_interval():
    return 0, args.nt

def _check_violation(obj, curr_objs):
    return any([cal_dist(obj, other_obj) < args.obj_min_gap for other_obj in curr_objs])
    
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

def _wrap_with_avoids(tmp_stl, tmp_objects: List, num_avoids=None):
    if num_avoids is None:
        num_avoids = np.random.choice(7, p=[0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1])  # 0~6 obstacles
    n_tries = 0
    avoids_stls = []
    tmp_goals = [xxx for xxx in tmp_objects]
    tmp_obstacles = []
    while n_tries < args.n_max_tries and len(tmp_obstacles) < num_avoids:
        n_tries += 1
        # if (n_tries < (args.n_max_tries//10) or n_tries % 5 == 0) and len(tmp_goals)>=2:
        #     choices = np.random.choice(len(tmp_goals), 2)
        #     object = _random_object(big=True, goals=[
        #         tmp_goals[choices[0]], 
        #         tmp_goals[choices[1]]
        #     ])
        # else:
        #     object = _random_object(big=True)  
        object = _random_object(big=True)  
        violation = _check_violation(object, tmp_objects)
        if violation == False:
            tc, td = _entire_interval()
            sub_stl = SimpleG(tc, td, SimpleNot(SimpleReach(len(tmp_objects), object=object)))
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


def collect_new_stl(stl_type_i, trial_i, random_seed):
    # types: (1) reach/stay, (2) and/or nested (3) sequential (4) until
    # each one can be combined with 0~k avoids
    np.random.seed(random_seed)
    curr_objects = []
    if stl_type_i == 0:   # reach / stay
        object = _random_object()
        curr_objects.append(object)
        ta, tb = _random_f_interval()
        final_stl = SimpleF(ta, tb, SimpleReach(0, object=object))
        final_stl = _wrap_with_random_stay(final_stl)
        final_stl, curr_objects, goals_indices, obstacles_indices = _wrap_with_avoids(final_stl, curr_objects)
                    
    elif stl_type_i == 1:   # and/or nested
        init_type = np.random.choice(["and", "or"], p=[0.5, 0.5])
        final_stl = SimpleListAnd([]) if init_type=="and" else SimpleListOr([])
        stack = [(0, init_type, final_stl)]
        curr_objects = []
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
                        object = _random_object()
                        violation = _check_violation(object, curr_objects)
                        if violation == False:
                            ta, tb = _random_f_interval()
                            reach_stl = SimpleF(ta, tb, SimpleReach(len(curr_objects), object=object))
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
        
        final_stl, curr_objects, goals_indices, obstacles_indices = _wrap_with_avoids(final_stl, curr_objects)
    
    elif stl_type_i == 2:   # sequential
        num_eles = np.random.randint(2, 5)  # choose 2~4
        prev_stl = None
        n_tries = 0
        while n_tries < args.n_max_tries and len(curr_objects) < num_eles:
            n_tries += 1
            object = _random_object()
            violation = _check_violation(object, curr_objects)
            if violation == False:
                ta, tb = _random_f_interval(small_a=True)
                sub_stl = SimpleF(ta, tb, SimpleReach(len(curr_objects), object=object))
                sub_stl = _wrap_with_random_stay(sub_stl)
                if prev_stl is not None:
                    # F(A) -> F(And(A, prev_stl))
                    # F(G(A)) -> F(And(G(A), prev_stl))
                    and_stl = SimpleAnd(left=sub_stl.children[0], right=prev_stl)
                    sub_stl.children = [and_stl]
                prev_stl = sub_stl
                curr_objects.append(object)
        final_stl = sub_stl
        final_stl, curr_objects, goals_indices, obstacles_indices = _wrap_with_avoids(final_stl, curr_objects)
            
    elif stl_type_i == 3:    # until
        until_patterns = [
            # 2
            ["A,B", "A"],
            # 3
            ["A,B", "B,C", "A"],
            ["A,B", "A,C", "A"],
            ["B,A", "C,A", "B", "C"],
            # 4
            ["A,B", "B,C", "C,D", "A"],
            ["A,B", "A,C", "A,D", "A"],
            ["A,B", "B,C", "B,D", "A"],
            ["A,B", "A,C", "C,D", "A"],
            ["A,B", "C,D", "A", "C"],
            ["B,A", "C,A", "D,A", "B", "C", "D"],
            # 5
            ["A,B", "B,C", "D,E", "A", "D"],
            ["A,B", "C,E", "D,E", "A", "C", "D"],
            # # 6
            # ["A,B", "B,C", "D,E", "E,F", "A", "D"],
            # ["A,B", "C,D", "E,F", "A", "C", "E"],
            # # 8
            # ["A,B", "C,D", "E,F", "G,H", "A", "C", "E", "G"],
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
                            object = _random_object()
                            violation = _check_violation(object, curr_objects)
                            if violation == False:
                                curr_objects.append(object)
                                break
                        assert violation==False
                        exist_objs[_ele] = len(curr_objects)-1
                tc, td = _default_g_interval()
                id0 = exist_objs[ele1]
                id1 = exist_objs[ele2]
                until_stl = SimpleUntil(tc, td, SimpleNot(SimpleReach(id0, object=curr_objects[id0])), SimpleReach(id1, object=curr_objects[id1]))
                until_stls.append(until_stl)
            else:
                id0 = exist_objs[ele]
                ta, tb = _entire_interval() # TODO here we consider a loose case
                reach_stl = SimpleF(ta, tb, SimpleReach(id0, object=curr_objects[id0])) 
                reach_stl = _wrap_with_random_stay(reach_stl)
                reach_stls.append(reach_stl)
        
        final_stl = SimpleListAnd(until_stls + reach_stls)
        final_stl, curr_objects, goals_indices, obstacles_indices = _wrap_with_avoids(final_stl, curr_objects)
        
    bloat = 0.5
    n_tries = 0
    while n_tries < args.n_max_tries:
        n_tries+=1
        ego_xy = np.random.uniform([args.x_min + bloat, args.y_min + bloat], [args.x_max - bloat, args.y_max - bloat])
        ego_state = ego_xy
        if all([cal_dist2(ego_xy, obj) > args.obj_min_gap for obj in curr_objects]):
            break
        
    if args.dynamics_type=="dubins":
        ego_th = np.random.uniform(-np.pi, np.pi)
        ego_v = np.random.uniform(0, args.v0_max)
        ego_state = np.array([ego_xy[0], ego_xy[1], ego_th, ego_v])

    # save_stl_to_file(final_stl, filepath="%s/stl_scene_%d_%d.txt"%(args.viz_dir, scene_type_i, trial_i), ego_xy=ego_xy)
    stl_np_lines = convert_stl_to_string(final_stl, numpy=True)
    record = {"ego":ego_state, "stl":stl_np_lines, "objects":curr_objects, "goals_indices":goals_indices, "obstacles_indices":obstacles_indices, "stl_seed":random_seed}
    return record, final_stl


def plot_tree(stl):
    color_list = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "brown", "gray", "pink", "royalblue", "lightgray", "lightgreen", "darkgreen", "salmon", "lightblue"]
    stl_color_list = ["orange", "yellow", "red", None, None, "cyan", "green", "purple", "gray"]
    stl_alpha = 1.0
    stl_r = 0.2
    stl_dx = 2.0
    stl_dy = 0.8
    init_width = 10.0
    stl_lw = 1.5
    stl_line_color = "darkblue"
    stl_line_alpha = 0.8
    stl_node_str_d={0:"&", 1:"|", 2:"¬", 3:"->", 4:"O", 5:"F", 6:"G", 7:"U", 8:"R"}
    
    ax = plt.gca()
    # plot the tree
    total_id = 0
    queue = [(0, stl, 0, init_width, 0, None)]
    coords = {0:[0,0]}
    while len(queue)!=0:
        stl_id, node, depth, width, order, father = queue[0]
        del queue[0]
        
        # plot self node
        # type_i = 0
        base_x = coords[stl_id][0]
        base_y = coords[stl_id][1]
        type_i = check_stl_type(node)
        circ = Circle([base_x, base_y], radius=stl_r, color=stl_color_list[type_i], alpha=stl_alpha, zorder=999)
        ax.add_patch(circ)
        
        # plot text
        plt.text(base_x-0.05, base_y-0.1, stl_node_str_d[type_i], fontsize=12, zorder=1005)
        
        n_child = len(node.children)
        for new_i, new_node in enumerate(node.children):
            # update child pos
            new_y = base_y - stl_dy
            if n_child==1:
                new_x = base_x
            else:
                M = 2 * n_child
                left_x = base_x - width / 2
                right_x = base_x + width / 2
                new_x = (M-new_i * 2 - 1) / M * left_x + (new_i * 2 + 1) / M * right_x
            
            # plot lines
            plt.plot([base_x, new_x], [base_y, new_y], linewidth=stl_lw, color=stl_line_color, alpha=stl_line_alpha, zorder=1)
            coords[total_id] = [new_x, new_y]
            
            # insert to queue
            queue.append([total_id, new_node, depth+1, width/n_child, new_i, node])
            total_id += 1
    
    plt.axis("scaled")
    plt.xlim(-init_width/2, init_width/2)
    plt.ylim(-(init_width-.5), 0.5)
    
    return

# TODO (viz function here)
def visualize_scene_and_stl(stl, stl_np_lines, objects, goals_indices, ego_xy, x_min, x_max, y_min, y_max, img_path):    
    obj_alpha = 0.5

    ego_size = 72
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    for obj_i, object in enumerate(objects):
        circ = Circle([object[0], object[1]], radius=object[2], color="gray" if obj_i not in goals_indices else "royalblue", alpha=obj_alpha)
        ax = plt.gca()
        ax.add_patch(circ)
    plt.scatter(ego_xy[0], ego_xy[1], color="green", marker="p", s=ego_size)
    plt.axis("scaled")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.subplot(1, 2, 2)
    plot_tree(stl)
    
    stl_dict = {}
    objects_d={}
    # print(stl_np_lines)
    real_stl = find_ap_in_lines(0, stl_dict=stl_dict, objects_d=objects_d, lines=stl_np_lines, numpy=True, real_stl=True, ap_mode="l2", until1=False)
    stl_string = real_stl.__str__()
    
    plt.suptitle(clip_str_window(stl_string, n=60))
    plt.tight_layout()
    utils.plt_save_close(img_path)

def visualize_scene_and_trajs(trajs_np, n_inits, stl_score, simple_stl, real_stl, objects_d, goals_indices, img_path):
    # plt.figure()
    n_rows = 2
    n_cols = 4
    fig = plt.figure(figsize=(11, 4))
    gs = fig.add_gridspec(2, 6)
    # gs = fig.add_gridspec(3, 6)
    fontsize = 10
    ego_markersize = 36
    for row_i in range(n_rows):
        for col_j in range(n_cols):
            index = row_i * n_cols + col_j + 1
            # plt.subplot(n_rows, n_cols, index)
            # f_ax1 = fig.add_subplot(gs[index//3, index%3])
            f_ax1 = fig.add_subplot(gs[(index-1)//4, (index-1)%4])
            ax = plt.gca()
            for obj_i, obj_keyid in enumerate(objects_d):
                obj_d = objects_d[obj_keyid]
                circ = Circle([obj_d["x"], obj_d["y"]], radius=obj_d["r"], color="royalblue" if obj_i in goals_indices else "gray", alpha=0.5)
                ax.add_patch(circ)
                plt.text(obj_d["x"], obj_d["y"], s="%d"%(obj_keyid), fontsize=fontsize)

            # plot ego state
            bs = (index-1) * n_inits
            be = (index-1+1) * n_inits
            for trajs_i in range(bs, be):
                plt.scatter(trajs_np[trajs_i, 0,0], trajs_np[trajs_i, 0,1], color="purple", s=ego_markersize, marker="X", zorder=1000)
                plt.plot(trajs_np[trajs_i, :,0], trajs_np[trajs_i, :,1], color="green" if stl_score[trajs_i]>0 else "red", alpha=0.3)
            plt.axis("scaled")
            # plt.axis("off")
            plt.xticks([])
            plt.yticks([])
            plt.xlim(args.x_min, args.x_max)
            plt.ylim(args.y_min, args.y_max)
    
    # f_ax1 = fig.add_subplot(gs[:, 3:6])
    f_ax1 = fig.add_subplot(gs[:, 4:6])
    ax = plt.gca()
    
    # plot STL tree
    stl = simple_stl
    stl_color_list = ["orange", "yellow", "salmon", None, None, "lightcyan", "palegreen", "turquoise", "cornflowerblue"]
    stl_alpha = 1.0
    stl_r = 0.3
    stl_dx = 2.0
    stl_dy = 0.8
    init_width = 10.0
    stl_lw = 1.5
    stl_line_color = "darkblue"
    stl_line_alpha = 0.8
    stl_node_str_d={0:"&", 1:"|", 2:"¬", 3:"->", 4:"O", 5:"F", 6:"G", 7:"U", 8:"R"}
    
    total_id = 0
    queue = [(0, stl, 0, init_width, 0, None)]
    coords = {0:[0,0]}
    while len(queue)!=0:
        stl_id, node, depth, width, order, father = queue[0]
        del queue[0]
        
        # plot self node
        # type_i = 0
        base_x = coords[stl_id][0]
        base_y = coords[stl_id][1]
        type_i = check_stl_type(node)
        if type_i==8:
            node_color = stl_color_list[type_i] if node.obj_id in goals_indices else "lightgray"
        else:
            node_color = stl_color_list[type_i]
        circ = Circle([base_x, base_y], radius=stl_r, color=node_color, alpha=stl_alpha, zorder=999)
        ax.add_patch(circ)
        
        # plot text
        # plt.text(base_x-0.05, base_y-0.1, stl_node_str_d[type_i], fontsize=fontsize, zorder=1005)
        if type_i==8:
            plt.text(base_x-0.25, base_y-0.1, "R%d"%(node.obj_id), fontsize=fontsize, zorder=1005)
        else:
            plt.text(base_x-0.15, base_y-0.1, stl_node_str_d[type_i], fontsize=fontsize, zorder=1005)
        
        n_child = len(node.children)
        for new_i, new_node in enumerate(node.children):
            # update child pos
            new_y = base_y - stl_dy
            if n_child==1:
                new_x = base_x
            else:
                M = 2 * n_child
                left_x = base_x - width / 2
                right_x = base_x + width / 2
                new_x = (M-new_i * 2 - 1) / M * left_x + (new_i * 2 + 1) / M * right_x
            
            # plot lines
            plt.plot([base_x, new_x], [base_y, new_y], linewidth=stl_lw, color=stl_line_color, alpha=stl_line_alpha, zorder=1)
            coords[total_id] = [new_x, new_y]
            
            # insert to queue
            queue.append([total_id, new_node, depth+1, width/n_child, new_i, node])
            total_id += 1
    
    plt.axis("scaled")
    plt.xlim(-init_width/2, init_width/2)
    plt.ylim(-(init_width-.5), 0.5)
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)     
    
    plt.suptitle("\n".join(textwrap.wrap(str(real_stl), width=72)))
    plt.tight_layout()
    utils.plt_save_close(img_path)

def clip_str_window(s, n):
    remain_s = s
    buffer = []
    while len(remain_s)>0:
        if len(remain_s)>n:
            buffer.append(remain_s[:n])
            remain_s = remain_s[n:]
        else:
            buffer.append(remain_s)
            remain_s = ""
            
    return "\n".join(buffer)


def main():
    utils.setup_exp_and_logger(args, test=args.test, dryrun=args.dryrun)
    
    NUM_CASES = 4
    all_cases = range(NUM_CASES)
    if args.clip_scenes is not None:
        NUM_CASES = len(args.clip_scenes)
        all_cases = args.clip_scenes
        remap = {scene_i: i for i, scene_i in enumerate(args.clip_scenes)}
        remap_inv = {v: k for k,v in remap.items()}
    
    bloat = 0.5
    x_min_safe = args.x_min + bloat
    x_max_safe = args.x_max - bloat
    y_min_safe = args.y_min + bloat
    y_max_safe = args.y_max - bloat
    global_md = utils.MeterDict()
    
    # np.savez("%s/data.npz"%(args.exp_dir_full), data=data_list)
    # args.data_dir = args.exp_dir_full
    # data_list = np.load("%s/data.npz"%(args.data_dir), allow_pickle=True)['data']
    
    success_cases = 0
    num_all_cases = 0
    
    succ_d = {kk:0 for kk in all_cases}
    num_all = {kk:0 for kk in all_cases}
    
    eta = utils.EtaEstimator(start_iter=0, end_iter=len(all_cases)*args.num_trials)
    
    if args.dynamics_type=="dubins":
        state_dim = 4
        action_dim = 2
        gen_traj_func = generate_trajectories_dubins
    else:
        state_dim = 2
        action_dim = 2
        gen_traj_func = generate_trajectories
    
    data_list = []
    os.makedirs("%s/cache_data"%(args.exp_dir_full))
    exp_idx = 0
    for trial_i in range(args.num_trials):
        for stl_type_i in all_cases:
            # if args.clip_scenes is not None and stl_type_i not in args.clip_scenes:
            #     continue
            eta.update()
            record, final_stl = collect_new_stl(stl_type_i, trial_i, random_seed = args.seed * 100000 + exp_idx)
            data_list.append(record)
            # visualization
            if trial_i < args.viz_max:
                img_path = "%s/viztree_%d_%04d.png"%(args.viz_dir, stl_type_i, trial_i)
                # visualize_scene_and_stl(final_stl, record['stl'], record["objects"], record["goals_indices"], record["ego"], args.x_min, args.x_max, args.y_min, args.y_max, img_path)
            
            if args.compute_traj:
                stl_dict = {}
                objects_d={}
                stl_record = record["stl"]
                simple_stl = find_ap_in_lines(0, stl_dict=stl_dict, objects_d=objects_d, lines=stl_record, numpy=True, ap_mode="l2", until1=False)
                curr_objects = get_current_objects_from_stl_lines(lines=stl_record, numpy=True)
                stl_dict = {}
                real_stl = find_ap_in_lines(0, stl_dict=stl_dict, objects_d=objects_d, lines=stl_record, numpy=True, real_stl=True, ap_mode="l2", until1=False)
                scene_type_i = stl_type_i
                
                # use gradient to solve it
                n_ego = 7
                n_inits = args.n_inits
                original_ego_state = record["ego"]
                
                # TODO
                goals_indices = record["goals_indices"]
                
                ego_state_list=[original_ego_state]
                n_tries = 0
                for ego_i in range(n_ego):
                    while n_tries < args.n_max_tries:
                        n_tries+=1
                        ego_xy = np.random.uniform([x_min_safe, y_min_safe], [x_max_safe, y_max_safe])
                        if args.dynamics_type=="dubins":
                            ego_th = np.random.uniform(-np.pi, np.pi)
                            ego_v = np.random.uniform(0, args.v0_max)
                            ego_state = np.array([ego_xy[0], ego_xy[1], ego_th, ego_v])
                        else:
                            ego_state = ego_xy
                            
                        if all([cal_dist2(ego_xy, obj) > args.obj_min_gap for obj in curr_objects]):
                            ego_state_list.append(ego_state)
                            break
                        
                while len(ego_state_list)!=n_ego+1:
                    ego_state_list.append(original_ego_state)
                n_ego_real = len(ego_state_list)
                
                ego_states = torch.from_numpy(np.stack(ego_state_list, axis=0)).float().to(device)
                ego_states = ego_states[:, None].repeat(1, n_inits, 1).reshape(n_ego_real, n_inits, state_dim)
                
                if args.dynamics_type=="dubins":
                    omegas = utils.uniform_tensor(-args.omega_max *.5, args.omega_max *.5, (n_ego_real, n_inits, args.nt, 1))
                    accels = utils.uniform_tensor(-args.accel_max *.5, args.accel_max *.5, (n_ego_real, n_inits, args.nt, 1))
                    us = torch.cat([omegas, accels], dim=-1).float().to(device).requires_grad_(True)
                else:
                    us = utils.uniform_tensor(-args.u_max *.5, args.u_max *.5, (n_ego_real, n_inits, args.nt, action_dim)).float().to(device).requires_grad_(True)
                
                md = utils.MeterDict()                
                optimizer = torch.optim.Adam([us], lr=args.trajopt_lr)
                for iter_i in range(args.trajopt_niters):
                    trajs = gen_traj_func(ego_states, us, dt=args.dt, v_max=args.v_max).reshape(n_ego_real * n_inits, args.nt+1, state_dim)
                    stl_score = real_stl(trajs, args.smoothing_factor)[:, :1]
                    acc = (stl_score>0).float()
                    acc_mean = torch.mean(acc)
                    loss_stl = torch.mean(torch.nn.ReLU()(args.stl_thres - stl_score)) * args.stl_weight
                    if args.dynamics_type=="dubins":
                        loss_reg = (torch.mean(torch.nn.ReLU()(us[...,0]**2 - args.omega_max**2)) * args.reg_weight + torch.mean(torch.nn.ReLU()(us[...,1]**2 - args.accel_max**2)) * args.reg_weight)/2
                        # loss_cost = torch.mean(us[...,0]**2) * args.cost_weight * 3 + torch.mean(us[...,1]**2) * args.cost_weight * 0.3 + torch.mean(torch.diff(us, dim=-2)**2) * args.cost_weight
                        loss_cost = torch.mean(us**2) * args.cost_weight * 0.1 # + torch.mean(torch.diff(us, dim=-2)**2) * args.cost_weight + torch.mean(torch.diff(torch.diff(us, dim=-2),dim=-2)**2) * args.cost_weight
                        loss_bdry_x = torch.mean(torch.nn.ReLU()(trajs[...,0]**2 - args.x_max**2)/(args.x_max**2)) * args.bdry_weight
                        loss_bdry_y = torch.mean(torch.nn.ReLU()(trajs[...,1]**2 - args.y_max**2)/(args.y_max**2)) * args.bdry_weight
                        loss_bdry = (loss_bdry_x + loss_bdry_y)/2
                    else:
                        loss_reg = torch.mean(torch.nn.ReLU()(us**2 - args.u_max**2)) * args.reg_weight
                        loss_cost = torch.mean(us**2) * args.cost_weight * 0.3 + torch.mean(torch.diff(us, dim=-2)**2) * args.cost_weight + torch.mean(torch.diff(torch.diff(us, dim=-2),dim=-2)**2) * args.cost_weight
                        loss_bdry = torch.mean(torch.nn.ReLU()(trajs**2 - args.x_max**2)/(args.x_max**2)) * args.bdry_weight
                    loss = loss_stl + loss_reg + loss_cost + loss_bdry
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    md.update("loss", loss.item())
                    md.update("loss_stl", loss_stl.item())
                    md.update("loss_reg", loss_reg.item())
                    md.update("loss_cost", loss_cost.item())
                    md.update("acc", acc_mean.item())
                                        
                    if (iter_i % 10 == 0 or iter_i ==args.trajopt_niters-1) and args.quiet==False:
                        print("Scene-%d id-%04d || Trajopt-Iter:%04d/%04d loss:%.3f(%.3f) stl:%.3f(%.3f) reg:%.3f(%.3f) cost:%.3f(%.3f) acc:%.3f(%.3f)"%(
                            scene_type_i, trial_i, iter_i, args.trajopt_niters, md["loss"], md("loss"), md["loss_stl"], md("loss_stl"), 
                            md["loss_reg"], md("loss_reg"), md["loss_cost"], md("loss_cost"), md["acc"], md("acc"), 
                        ))
                    
                    if iter_i==args.trajopt_niters-1:
                        global_md.update("acc", acc_mean.item())
                        if args.quiet==False:
                            print("(%04d/%04d) Accuracy:%.3f(%.3f)"%(exp_idx, 
                                    args.num_trials * NUM_CASES, 
                                    global_md["acc"], global_md("acc")))
                
                if acc_mean.item()>0:
                    succ_d[stl_type_i] += 1
                    success_cases += 1
                num_all[stl_type_i] += 1
                num_all_cases +=1
                
                print("%s Acc/Case: %d/%d(%.3f) {%s} Accuracy:%.3f(%.3f)  dT:%s  Elapsed:%s  ETA:%s"%(
                    args.exp_dir_full.split("/")[-1], success_cases, num_all_cases, success_cases/num_all_cases,
                    ", ".join(["%d:(%.3f)"%(type_, succ_d[type_]/np.clip(num_all[type_],1e-4,1e7)) for type_ in all_cases]),
                    global_md["acc"], global_md("acc"),
                    eta.interval_str(), eta.elapsed_str(), eta.eta_str(),
                    ))
                
                trajs_np = utils.to_np(trajs)
                # visualize
                if trial_i < args.viz_max:
                    img_path = "%s/img_%d_%05d.png"%(args.viz_dir, scene_type_i, trial_i)
                    visualize_scene_and_trajs(trajs_np, n_inits, stl_score, simple_stl, real_stl, objects_d, goals_indices, img_path)
                
                # save trajs as well as their scores
                record["stl_type_i"] = stl_type_i
                record["trial_i"] = trial_i
                record["score"] = utils.to_np(stl_score)
                record["us"] = utils.to_np(us)
                record["state"] = trajs_np[..., 0, :]
                # record["trajs"] = trajs_np[..., :, :] # save space
            
            if exp_idx % args.save_freq == 0 or exp_idx == args.num_trials * NUM_CASES-1:
                np.savez("%s/data.npz"%(args.exp_dir_full), data=data_list)
            if trial_i==100:
                np.savez("%s/data100.npz"%(args.exp_dir_full), data=data_list)    
            exp_idx += 1
        np.savez("%s/data.npz"%(args.exp_dir_full), data=data_list)            
    return

def batch_worker(job_i, seed):
    global args
    import copy
    # Create a local copy of args for this process
    args_local = copy.deepcopy(args)
    args_local.seed = seed
    args_local.job_id = job_i
    args = args_local  # Update the global args
    print("Run with", args.seed)
    if job_i>=2:
        sys.stdout = open(os.devnull, 'w')
    t1 = time.time()
    main()  # Call main() with updated args
    t2 = time.time()
    print("Finished in %.3f seconds"%(t2-t1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--seed", type=int, default=1007)
    add("--exp_name", '-e', type=str, default="old_genscene")
    add("--gpus", type=str, default="0")
    add("--cpu", action='store_true', default=False)
    add("--test", action='store_true', default=False)
    add("--dryrun", action='store_true', default=False)
    add("--num_trials", type=int, default=100)
    add("--nt", type=int, default=64)
    
    add("--data_dir", type=str, default="0")
    add("--collect_stl", action='store_true', default=False)
    add("--compute_traj", action='store_true', default=False)
    add("--trajopt_niters", type=int, default=150)
    add("--trajopt_lr", type=float, default=1e-1)
    add("--dt", type=float, default=0.5)
    add("--smoothing_factor", type=float, default=10.0)
    
    add("--stl_thres", type=float, default=0.5)
    add("--stl_weight", type=float, default=1.0)
    add("--reg_weight", type=float, default=1.0)
    add("--cost_weight", type=float, default=20.0)
    add("--bdry_weight", type=float, default=5)
    
    add("--new_specs", action='store_true', default=False)
    add("--waypoints", action='store_true', default=False)
    add("--clip_scenes", type=int, nargs="+", default=None)
    add("--viz_max", type=int, default=3)
    add("--save_freq", type=int, default=100)
    
    add("--u_max", type=float, default=1.0)
    
    add("--r_min", type=float, default=0.5)
    add("--r_max", type=float, default=1.5)
    add("--x_min", type=float, default=-5.0)
    add("--x_max", type=float, default=5.0)
    add("--y_min", type=float, default=-5.0)
    add("--y_max", type=float, default=5.0)
    add("--bloat_ratio", type=float, default=1.05)
    add("--obj_min_gap", type=float, default=1.05)
    add("--n_max_tries", type=int, default=1000)
    add("--n_inits", type=int, default=8)
    
    add("--quiet", action='store_true', default=False)
    add("--batch", type=int, default=None)
    
    add("--dynamics_type", type=str, choices=["simple", "dubins"], default="simple")
    add("--v0_max", type=float, default=1.0)
    add("--omega_max", type=float, default=1.0)
    add("--accel_max", type=float, default=1.0)
    add("--v_max", type=float, default=2.0)
    
    
    args = parser.parse_args()
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda:0"
    
    if args.batch is not None:
        t1 = time.time()
        from multiprocessing import Process
        processes = []
        seed_start = args.seed
        for i in range(args.batch): # Generate a unique seed for each process
            p = Process(target=batch_worker, args=(i, seed_start + i))
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()
        t2 = time.time()
        print("All parallel runs are completed in %.3f seconds"%(t2-t1))
        
    else:
        t1 = time.time()
        main()
        t2 = time.time()
        print("Finished in %.3f seconds"%(t2-t1))