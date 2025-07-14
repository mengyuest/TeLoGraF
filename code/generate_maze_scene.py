import os
import math
import heapq
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils
import random
import torch
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
from stable_baselines3 import SAC
from itertools import product
from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle
import traceback

import itertools
POLICY_FILE_PATH = "./ant_goal_reaching.ckpt"

from generate_scene_v1 import SimpleAnd, SimpleOr, SimpleListAnd, SimpleListOr, SimpleF, SimpleG, SimpleReach, SimpleUntil, SimpleNot, \
    convert_stl_to_string, find_ap_in_lines

class MockMaze:
    def __init__(self, maze):
        self.maze_map = maze
        self.map_length = len(self.maze_map)
        self.map_width = len(self.maze_map[0])

def find_all_dependencies(list_deps):
    dep_list_dict = {}
    for dep in list_deps:
        for key in dep:
            if key not in dep_list_dict:
                dep_list_dict[key] = set()
            for ele in dep[key]:
                dep_list_dict[key].add(ele)
    memory = {}
    for key in dep_list_dict:
        ret = find_all_dependencies_inner(0, key, dep_list_dict, memory)
        if ret is None:
            break
    return memory

def find_all_dependencies_inner(stack_i, curr_val, dep_list_dict, memory):
    if stack_i>100:
        print("Likely has a cycle in the dependencies, abort...")
        return None
    dependencies = set()
    if curr_val in dep_list_dict:
        for depend_val in dep_list_dict[curr_val]:
            dependencies.add(depend_val)
            if depend_val in memory:
                dep_dependencies = memory[depend_val]
            else:
                dep_dependencies = find_all_dependencies_inner(stack_i+1, depend_val, dep_list_dict, memory)
                if dep_dependencies is None:
                    return None
                memory[depend_val] = dep_dependencies
            dependencies = dependencies.union(dep_dependencies)
    memory[curr_val] = dependencies
    return dependencies

def get_fwdPathList_from_planned_path(planned_path):
    visited = set()
    fwdPath = {}
    fwdPathList=[fwdPath]
    flat_path_list=[]
    for i in range(len(planned_path)-1):
        visited.add(planned_path[i])
        if planned_path[i+1] in visited:
            flatPath = []
            for key in fwdPath:
                flatPath.append(key)
            flatPath.append(fwdPath[list(fwdPath.keys())[-1]]) 
            flat_path_list.append(flatPath)
            
            visited = set()
            fwdPath = {}
            fwdPathList.append(fwdPath)
        fwdPath[planned_path[i]] = planned_path[i+1]
    
    flatPath = []
    for key in fwdPath:
        flatPath.append(key)
    flatPath.append(fwdPath[list(fwdPath.keys())[-1]])
    flat_path_list.append(flatPath)
    
    return fwdPathList, flat_path_list

class AStar:
    """A* Search Algorithm."""
    def __init__(self, maze, add_noise_heuristics=False):
        self.maze = maze
        self.UP = UP = 0
        self.DOWN = DOWN = 1
        self.LEFT = LEFT = 2
        self.RIGHT = RIGHT = 3
        self.add_noise_heuristics = add_noise_heuristics
        self.EXPLORATION_ACTIONS = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0)}

    def generate_path_with_deps(self, current_pos, desired_pos, blocks=None, banned_first_actions=None, debug=False, rois=None, list_deps=None, stoc_number=0):
        start = tuple(current_pos)
        goal = tuple(desired_pos)
        frontier = []
        heapq.heappush(frontier, (0, start, frozenset(), [start]))  # (priority, cell, visited_vals)
        g_cost = {(start,frozenset()): 0}
        n_steps=0
        
        assert rois is not None
        assert list_deps is not None
        
        action_cands = self.EXPLORATION_ACTIONS.values()
        if stoc_number!=0:
            # (0) 1, 2, 3, 4
            # (10) 2, 4, 1, 3
            # (13) 3, 1, 4, 2
            # (23) 4, 3, 2, 1
            cantor_list = [0, 10, 13, 23]
            action_cands = list(itertools.permutations(action_cands))[cantor_list[stoc_number % 4]]
        
        memory = find_all_dependencies(list_deps)
        possible_paths = []
        while frontier:
            _, current_cell, visited_cell_ids, planned_path = heapq.heappop(frontier)
            n_steps+=1
            if current_cell == goal:
                possible_paths.append(planned_path)
                break
            
            for action in action_cands:
                # special treatment for diversity
                if n_steps==1 and banned_first_actions is not None and action in banned_first_actions:
                    continue
                new_cell = tuple(map(lambda i, j: int(i + j), current_cell, action))
                
                # collision
                if not self._check_valid_cell(new_cell) or (blocks is not None and new_cell in blocks):
                    continue
                
                # dependencies
                # TODO (I think we can run this one time at init)
                if new_cell in rois:
                    new_cell_id = rois[new_cell]
                    if new_cell_id in memory:
                        dep_cell_ids = memory[new_cell_id]
                        broken_dependency = False
                        for dep_cell_id in dep_cell_ids:
                            if dep_cell_id not in visited_cell_ids:
                                broken_dependency=True
                                break
                        if broken_dependency:
                            continue
                else:
                    new_cell_id = None
                
                new_cost = g_cost[(current_cell,visited_cell_ids)] + 1  # Uniform cost (each move = 1)
                
                new_visited_cell_ids = set(visited_cell_ids)
                if new_cell_id is not None:
                    new_visited_cell_ids.add(new_cell_id)
                new_visited_cell_ids = frozenset(new_visited_cell_ids)
                
                if (new_cell, new_visited_cell_ids) not in g_cost or new_cost < g_cost[(new_cell,new_visited_cell_ids)]:
                    g_cost[(new_cell, new_visited_cell_ids)] = new_cost
                    priority = new_cost + self._heuristic_dep(new_cell, goal, new_visited_cell_ids, stoc=stoc_number!=0)
                    new_planned_path = list(planned_path)
                    new_planned_path.append(new_cell)
                    heapq.heappush(frontier, (priority, new_cell, new_visited_cell_ids, new_planned_path))

        if len(possible_paths)==0:
            return None  # No path found

        fwdPathList, flat_path_list = get_fwdPathList_from_planned_path(planned_path)
        return fwdPathList, flat_path_list, planned_path
            

    def _heuristic_dep(self, cell, goal, new_visited_cell_ids, stoc=False):
        """Euclidean distance heuristic."""
        if self.add_noise_heuristics:
            return math.sqrt((goal[0] - cell[0]) ** 2 + (goal[1] - cell[1]) ** 2) + np.random.uniform(0, 50) + len(new_visited_cell_ids) * 10
        elif stoc:
            return math.sqrt((goal[0] - cell[0]) ** 2 + (goal[1] - cell[1]) ** 2) + np.random.uniform(0, 25) + len(new_visited_cell_ids) * 10
        else:
            return math.sqrt((goal[0] - cell[0]) ** 2 + (goal[1] - cell[1]) ** 2) + len(new_visited_cell_ids) * 10

    def generate_path(self, current_pos, desired_pos, blocks=None, banned_first_actions=None, debug=False, with_flat=False, stoc_number=0):
        start = tuple(current_pos)
        goal = tuple(desired_pos)
        
        frontier = []
        heapq.heappush(frontier, (0, start))  # (priority, cell)
        
        g_cost = {start: 0}
        came_from = {}
        
        action_cands = self.EXPLORATION_ACTIONS.values()
        if stoc_number!=0:
            # (0) 1, 2, 3, 4
            # (10) 2, 4, 1, 3
            # (13) 3, 1, 4, 2
            # (23) 4, 3, 2, 1
            cantor_list = [0, 10, 13, 23]
            action_cands = list(itertools.permutations(action_cands))[cantor_list[stoc_number % 4]]
        n_steps=0
        while frontier:
            _, current_cell = heapq.heappop(frontier)
            n_steps+=1
            
            if current_cell == goal:
                break

            for action in action_cands:
                if n_steps==1 and banned_first_actions is not None and action in banned_first_actions:
                    continue
                new_cell = tuple(map(lambda i, j: int(i + j), current_cell, action))

                if not self._check_valid_cell(new_cell) or (blocks is not None and new_cell in blocks):
                    continue

                new_cost = g_cost[current_cell] + 1  # Uniform cost (each move = 1)
                if new_cell not in g_cost or new_cost < g_cost[new_cell]:
                    g_cost[new_cell] = new_cost
                    priority = new_cost + self._heuristic(new_cell, goal, stoc=stoc_number!=0)
                    heapq.heappush(frontier, (priority, new_cell))
                    came_from[new_cell] = current_cell

        if goal not in came_from:
            if with_flat:
                return None, None  # No path found
            else:
                return None

        # Reconstruct path
        fwdPath = {}
        cell = goal
        while cell != start:
            fwdPath[came_from[cell]] = cell
            cell = came_from[cell]
        if debug:
            return fwdPath, n_steps
        else:
            if with_flat:
                return fwdPath, get_path(fwdPath, current_pos, desired_pos)
            else:
                return fwdPath

    def _heuristic(self, cell, goal, stoc=False):
        """Euclidean distance heuristic."""
        if self.add_noise_heuristics:
            return math.sqrt((goal[0] - cell[0]) ** 2 + (goal[1] - cell[1]) ** 2) + np.random.uniform(0, 50)
        elif stoc:
            # print("ASD", math.sqrt((goal[0] - cell[0]) ** 2 + (goal[1] - cell[1]) ** 2) + np.random.uniform(0, 250))
            return math.sqrt((goal[0] - cell[0]) ** 2 + (goal[1] - cell[1]) ** 2) + np.random.uniform(0, 25)
        else:
            return math.sqrt((goal[0] - cell[0]) ** 2 + (goal[1] - cell[1]) ** 2)

    def _heuristic2(self, cell, goal):
        """Euclidean distance heuristic."""
        return abs(goal[0] - cell[0]) * 5 + abs(goal[1] - cell[1])
    
    def _heuristic3(self, cell, goal):
        """Euclidean distance heuristic."""
        return abs(goal[0] - cell[0]) + abs(goal[1] - cell[1]) * 10

    def _check_valid_cell(self, cell):
        """Check if the cell is valid (within bounds & not an obstacle)."""
        if cell[0] >= self.maze.map_length or cell[0] < 0:
            return False
        if cell[1] >= self.maze.map_width or cell[1] < 0:
            return False
        if self.maze.maze_map[cell[0]][cell[1]] == 1:  # Obstacle
            return False
        return True


class WaypointController:
    """Generic agent controller to follow waypoints in the maze.
    Inspired by https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/pointmaze/waypoint_controller.py
    """
    def __init__(self, maze, model_callback, waypoint_threshold=0.45):
        self.global_target_xy = np.empty(2)
        self.maze = maze
        self.maze_solver = AStar(maze=self.maze)
        self.model_callback = model_callback
        self.waypoint_threshold = waypoint_threshold
        self.waypoint_targets = None

    def reset(self, obs, waypoint_target, desired_goal=None):
        if desired_goal is None:
            desired_goal = obs["desired_goal"]
        
        achived_goal_cell = tuple(self.maze.cell_xy_to_rowcol(obs["achieved_goal"]))
        
        self.global_target_id = tuple(self.maze.cell_xy_to_rowcol(desired_goal))
        self.global_target_xy = desired_goal
        self.waypoint_targets = waypoint_target
        
        if self.waypoint_targets:
            self.current_control_target_id = self.waypoint_targets[achived_goal_cell]
            # If target is global goal go directly to goal position
            if self.current_control_target_id == self.global_target_id:
                self.current_control_target_xy = desired_goal
            else:
                self.current_control_target_xy = self.maze.cell_rowcol_to_xy(
                    np.array(self.current_control_target_id)
                ) - np.random.uniform(size=(2,)) * 0.0
        else:
            self.waypoint_targets[self.current_control_target_id] = self.current_control_target_id
            self.current_control_target_id = self.global_target_id
            self.current_control_target_xy = self.global_target_xy
        
    def policy(self, obs, desired_goal=None, noise_free=True, action_noise=None):
        action = self.compute_action_fixed(obs, desired_goal=desired_goal)
        if not noise_free:
            action += action_noise * np.random.randn(*action.shape)
        action = np.clip(action, -1.0, 1.0)
        return action

    def compute_action_fixed(self, obs, desired_goal=None):    
        if desired_goal is None:
            desired_goal = obs['desired_goal']
        # Check if we need to go to the next waypoint
        dist = np.linalg.norm(self.current_control_target_xy - obs["achieved_goal"])        
                
        if (dist <= self.waypoint_threshold
            and self.current_control_target_id != self.global_target_id
        ):  
            # print(self.current_control_target_id, self.global_target_id, self.waypoint_targets)
            self.current_control_target_id = self.waypoint_targets[self.current_control_target_id]
            # If target is global goal go directly to goal position
            if self.current_control_target_id == self.global_target_id:
                self.current_control_target_xy = desired_goal
            else:
                self.current_control_target_xy = (
                    self.maze.cell_rowcol_to_xy(
                        np.array(self.current_control_target_id)
                    )
                    - np.random.uniform(size=(2,)) * 0.1
                )  
        action = self.model_callback(obs, self.current_control_target_xy)
        return action


def get_path(path, start, goal):
    path_flat = []
    curr = start
    if path is None:
        return None
    while curr != goal:
        path_flat.append([curr[0], curr[1]])
        curr = path[curr]
    path_flat.append(goal)
    return path_flat


def gen_hard_maze():
    C = 0
    maze = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, C, 0, 0, 0, 1, C, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, C, 0, 1, 0, 0, C, 1],
            [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, C, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, C, 0, C, 1, 0, C, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    return maze


def wrap_maze_obs(obs, waypoint_xy):
    """Wrap the maze obs into one suitable for GoalReachAnt."""
    goal_direction = (waypoint_xy - obs["achieved_goal"])
    observation = np.concatenate([obs["observation"], goal_direction])
    return observation


def generate_assignments_stl(node):
    """Recursively generate all feasible assignments for the AND/OR tree."""
    if isinstance(node, SimpleF):
        return [{node.children[0].obj_id}]  # A single assignment set with one object

    child_assignments = [generate_assignments_stl(child) for child in node.children]

    if isinstance(node, SimpleAnd) or isinstance(node, SimpleListAnd):
        # Cartesian product: must satisfy all children's assignments together
        return [set().union(*combo) for combo in product(*child_assignments)]

    elif isinstance(node, SimpleOr) or isinstance(node, SimpleListOr):
        # OR requires at least one child to be satisfied → Collect all child results
        result = []
        for assignments in child_assignments:
            result.extend(assignments)  # Just take any valid child solution
        return result

    return []  # Should not reach here

def plan_for_multi_goals(solver, current_pos, desired_pos_list, max_path_len=12, stoc_number=0, banned_first_actions=None, ordered=False):
    num_seqs = len(desired_pos_list)
    poss_paths = []
    stack = [(current_pos, {}, [current_pos], [], [])]
    while len(stack)>0:
        node = stack[-1]
        del stack[-1]
        curr_pos, visited, curr_path, pathd_list, path_flat_list = node
        
        if len(visited)==num_seqs and len(curr_path) <= max_path_len:
            new_curr_path = [path_item for path_item in curr_path]
            new_pathd_list = [path_item for path_item in pathd_list]
            new_path_flat_list = [path_item for path_item in path_flat_list]
            poss_paths.append((new_curr_path, new_pathd_list, new_path_flat_list))

        for seq_i in range(num_seqs):
            if ordered:
                if seq_i != len(visited):
                    continue
            if seq_i not in visited:
                new_visited = set(visited)
                new_visited.add(seq_i)
                sub_path = solver.generate_path(curr_pos, desired_pos_list[seq_i], stoc_number=stoc_number, banned_first_actions=banned_first_actions if len(visited)==0 else [])
                if sub_path is not None:
                    sub_path_flat = get_path(sub_path, curr_pos, desired_pos_list[seq_i])
                    new_pathd_list = [path_item for path_item in pathd_list]
                    new_pathd_list.append(sub_path)
                    
                    new_path_flat_list = [path_item for path_item in path_flat_list]
                    new_path_flat_list.append(sub_path_flat)
                    for seq_j in range(num_seqs):
                        if seq_j not in new_visited and tuple(desired_pos_list[seq_j]) in sub_path_flat:
                            new_visited.add(seq_j)
                    
                    stack.append([desired_pos_list[seq_i], new_visited, curr_path + [tuple(xx) for xx in sub_path_flat[1:]], new_pathd_list, new_path_flat_list])
    
    return poss_paths


def viz_biplot_maze(env, trajs_cat, fname, curr=None, goal_list=None, rois=None):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    rgb_img = env.unwrapped.render()
    plt.imshow(rgb_img, extent=[0, 100, 0, 100])
    plt.subplot(1, 2, 2)
    if trajs_cat is not None:
        plt.plot(trajs_cat[:, 0], trajs_cat[:, 1], color="royalblue", alpha=0.5, linewidth=2.0)
    if curr is not None:
        a = env.unwrapped.maze.maze_size_scaling
        x, y = env.unwrapped.maze.cell_rowcol_to_xy(curr)
        rect = Rectangle([x-a/2 ,y-a/2], a, a, color="royalblue", alpha=0.5)
        ax = plt.gca()
        ax.add_patch(rect)
    if goal_list is not None:
        for goal in goal_list:
            a = env.unwrapped.maze.maze_size_scaling
            x, y = env.unwrapped.maze.cell_rowcol_to_xy(goal)
            rect = Rectangle([x-a/2 ,y-a/2], a, a, color="green", alpha=0.5)
            ax = plt.gca()
            ax.add_patch(rect)
    if rois is not None:
        for key in rois:
            loc = key
            txt = rois[key]
            a = env.unwrapped.maze.maze_size_scaling
            x, y = env.unwrapped.maze.cell_rowcol_to_xy(loc)
            rect = Rectangle([x-a/2 ,y-a/2], a, a, color="orange" if "k" in txt else "red", alpha=0.5)
            plt.text(x, y, s=txt)
            ax = plt.gca()
            ax.add_patch(rect)
            
    plt.axis("scaled")
    plt.xlim(-env.unwrapped.maze._x_map_center, env.unwrapped.maze._x_map_center)
    plt.ylim(-env.unwrapped.maze._y_map_center, env.unwrapped.maze._y_map_center)
    plt.savefig(fname)
    plt.close()


def gen_trajs_from_waypoints(seed_i, type_i, env, controller, seed, init_cell, possd_list, path_list, nt, wpt_thres, sat_thres, goal_list, rois, fname_pattern):
    goal_cell =  path_list[-1][-1]
    current_goal_xy = rowcol_to_xy(env, path_list[0][-1])
    obs, info = env.reset(seed=seed, options={"reset_cell":init_cell, "goal_cell": goal_cell})
    controller.reset(obs, possd_list[0], desired_goal=current_goal_xy)
    trajs = [obs["achieved_goal"]]
    seg_i = 0
    action_list = []
    obs_list = [obs["observation"]]  # dict_keys(['observation', 'achieved_goal', 'desired_goal'])
    not_in_goal_yet = True
    for ti in range(nt):
        if near(obs, current_goal_xy, wpt_thres):
            if seg_i!=len(possd_list)-1:
                seg_i+=1
                current_goal_xy = rowcol_to_xy(env, path_list[seg_i][-1])
                controller.reset(obs, possd_list[seg_i], desired_goal=current_goal_xy)     
            else:
                if not_in_goal_yet:
                    not_in_goal_yet = False
                    total_len = sum([len(pdd) for pdd in possd_list])
                    # print("First reach at %d/%d path_len:%d  pace:%.3f/step"%(ti, nt, total_len, ti/total_len))
                                        
                                    
        action = controller.policy(obs, desired_goal=current_goal_xy)
        obs, _, _, _, _ = env.step(action)
        trajs.append(obs["achieved_goal"])
        action_list.append(action)
        obs_list.append(obs["observation"])
        if ti==nt-1 and args.no_viz==False and (seed_i % args.viz_freq==0 or seed_i == args.num_trials-1):
            viz_biplot_maze(env, np.stack(trajs, axis=0), fname_pattern%(ti), init_cell, goal_list=goal_list, rois=rois)
    
    # if seg_i==len(possd_list)-1 and near(obs, rowcol_to_xy(env, goal_cell), sat_thres):
    if seg_i==len(possd_list)-1 and near(obs, rowcol_to_xy(env, goal_cell), sat_thres):
        # print(seg_i, len(possd_list)-1)
        is_sat = True
        # print("Succeeded", seed)
    else:
        is_sat = False
    
    record = {}
    record["stl_type_i"] = type_i
    record["stl_seed"] = seed
    record["init_cell"] = init_cell
    record["goal_cell"] = goal_cell
    record["ego"] = rowcol_to_xy(env, init_cell)
    record["goals_indices"] = []
    record["obstacles_indices"] = []
    if args.dynamics_type=="antmaze":
        record["trajs"] = np.stack(trajs, axis=0)  # (T+1, 2)
    record["obs"] = np.stack(obs_list, axis=0)  # (T+1, 27) / (T+1, 4)
    record["actions"] = np.stack(action_list, axis=0)  # (T, 8) / (T+1, 2)
    return record, is_sat

def rowcol_to_xy(env, rowcol):
    return env.unwrapped.maze.cell_rowcol_to_xy(rowcol)

def xy_to_rowcol(env, xy):
    return env.unwrapped.maze.cell_xy_to_rowcol(xy)

def near(obs, point, thres):
    return np.linalg.norm(obs["achieved_goal"] - point) < thres

def _check_valid(maze, cell):
    """Check if the cell is valid (within bounds & not an obstacle)."""
    if cell[0] >= len(maze) or cell[0] < 0:
        return False
    if cell[1] >= len(maze[0]) or cell[1] < 0:
        return False
    if maze[cell[0]][cell[1]] == 1:  # Obstacle
        return False
    return True

def random_walk(maze, start, max_step, visited, type=0, goal=None):
    queue = [(0, start, [start])]
    actions = [[-1,0],[1,0],[0,1],[0,-1]]
    tmp_visited = {}
    while len(queue)>0:
        step, curr, existing_path = queue[0]
        del queue[0]
        if curr not in tmp_visited:
            tmp_visited[curr] = step
        elif tmp_visited[curr] < step:
            if len(queue)==0:
                return step, curr, existing_path
            else: 
                continue
        if goal is not None:
            if curr == goal:
                return step, curr, existing_path
        else:
            if len(existing_path)==max_step:
                if tuple(curr) in visited:
                    continue
                else:
                    return step, curr, existing_path
        random.shuffle(actions)
        for act in actions:
            new_cell = tuple((curr[0]+act[0], curr[1]+act[1]))
            if new_cell not in existing_path and _check_valid(maze, new_cell):
                new_path = [xxx for xxx in existing_path]
                new_path.append(new_cell)
                queue.append([step+1, new_cell, new_path])
    return None, None, None
    

def random_free_cell(maze, visited=None):
    free_cells = (1-maze).nonzero()  # (idx0,...), (idx1,...)
    free_cells = set([tuple((xxx,yyy)) for xxx,yyy in zip(free_cells[0], free_cells[1])])    
    if visited is not None:
        free_cells = free_cells - visited
    free_cells = list(free_cells)
    rand_idx = np.random.choice(len(free_cells))
    return free_cells[rand_idx]

def check_in_region(trajs, roi_xy, inside_thres):
    roi_xy_np = np.array(roi_xy[:2]).reshape(1, 2)
    dist = np.linalg.norm(trajs-roi_xy_np, 1, axis=-1)
    return (dist < inside_thres) * 1.0

def cell_to_object(env, cell):
    x, y = rowcol_to_xy(env, cell)
    return [x, y, 1.0]

def get_subgoal_intervals_from_trajs(tmp_cache, curr_objects, mode="min", stl_type_i=None):
    if args.dynamics_type=="antmaze":
        inside_thres = 1.0
    else:
        inside_thres = 0.42
    intervals_d = {objid:[] for objid in range(len(curr_objects))}
    if stl_type_i==2:
        final_intes_list=[]
        for ri, record in enumerate(tmp_cache):
            if args.dynamics_type=="antmaze":
                xys = record["trajs"]
            else:
                xys = record["obs"][..., :2]
            intes=[]
            for obj_id in intervals_d:
                subgoal = curr_objects[obj_id]
                in_goal_region = check_in_region(xys, subgoal[:2], inside_thres=inside_thres)
                in_indices = in_goal_region.nonzero()[0]
                # print(ri, obj_id, in_indices)
                seg_check = in_indices[1:]!=in_indices[:-1]+1
                splits = seg_check.nonzero()[0]
                prev_split = 0
                if len(splits)>0:
                    for split_i in splits:
                        intes.append([in_indices[prev_split], in_indices[split_i], obj_id])
                        prev_split = split_i+1
                    intes.append([in_indices[prev_split], in_indices[len(in_indices)-1], obj_id])
                elif len(in_indices)>0:
                    intes.append([in_indices[0], in_indices[len(in_indices)-1], obj_id])
            intes = sorted(intes, key=lambda x:x[0])
            ordered_obj_id = sorted(np.unique([inte[-1] for inte in intes]), reverse=True)
            # print('intes', intes, ordered_obj_id)
            # linear scan, reversed order pickup, 2->1->0
            ptr=0
            final_intes={}
            for inte in intes:
                if ptr<len(ordered_obj_id) and inte[-1]==ordered_obj_id[ptr]:
                    ptr+=1
                    final_intes[inte[2]]=inte[:2]
            final_intes_list.append(final_intes)

    for obj_id in intervals_d:
        subgoal = curr_objects[obj_id]
        for ri, record in enumerate(tmp_cache):
            if args.dynamics_type=="antmaze":
                xys = record["trajs"]
            else:
                xys = record["obs"][..., :2]
            if stl_type_i==2:
                in_indices = final_intes_list[ri][obj_id]
            else:
                in_goal_region = check_in_region(xys, subgoal[:2], inside_thres=inside_thres)
                in_indices = in_goal_region.nonzero()[0]
            if stl_type_i==1 and len(in_indices)>0:
                seg_check = in_indices[1:]!=in_indices[:-1]+1
                splits = seg_check.nonzero()[0]
                prev_split = 0
                if len(splits)>0:
                    in_indices = [in_indices[prev_split], in_indices[splits[0]]]
                elif len(in_indices)>0:
                    in_indices = [in_indices[0], in_indices[-1]]
            
            if len(in_indices)>0:
                start_idx = in_indices[0]
                end_idx = in_indices[-1]
                # print(obj_id, start_idx, end_idx, intervals_d[obj_id])
                if mode=="min":
                    if len(intervals_d[obj_id])==0:
                        intervals_d[obj_id]=[start_idx, end_idx]
                    else:
                        if intervals_d[obj_id][0] < start_idx:
                            intervals_d[obj_id][0] = start_idx
                        if intervals_d[obj_id][1] > end_idx:
                            intervals_d[obj_id][1] = end_idx
                elif mode=="all":
                    intervals_d[obj_id].append([start_idx, end_idx])
                else:
                    raise NotImplementedError
            
        # TODO exception
        if len(intervals_d[obj_id])==0:
            if mode=="min":
                intervals_d[obj_id] = []
            elif mode=="all":
                intervals_d[obj_id] = []
            else:
                raise NotImplementedError
    return intervals_d

def update_f_oper_interval_with_g(stl_f, interval):
    # now we find the interval_d, randomly sample a small region
    # randomly add a G operator
    if interval[0]>=interval[1]-1:
        stl_f.ts = -1
        stl_f.te = -1
    else:
        f_ts, f_te = sorted(np.random.choice(range(interval[0], interval[1]+1), 2, replace=False))
        stl_f.ts = f_ts
        stl_f.te = f_te
        if np.random.rand()>0.5 and f_te - f_ts > 1: # add G
            g_te = np.random.randint(1, f_te - f_ts)
            g_stl = SimpleG(0, g_te, ap=stl_f.children[0])
            stl_f.children = [g_stl]      

def collect_stl_records(data_list, tmp_cache, final_stl, curr_objects):
    for record in tmp_cache:
        stl_np_lines = convert_stl_to_string(final_stl, numpy=True)
        record["stl"] = stl_np_lines
        record["objects"] = curr_objects
        data_list.append(record)

def random_free_cells(maze, n_points):
    available_indices = (1-np.array(maze.maze_map)).nonzero()
    index = np.random.choice(len(available_indices[0]), n_points, replace=False)
    return [(available_indices[0][index[iii]], available_indices[1][index[iii]]) for iii in range(n_points)]

def rand_sample_loc_stl(stl_type_i, solver, env, maze, total_timesteps):
    stl_part_d = {}
    curr_cells = []
    curr_objects = []
    assignments = None
    poss_paths = None 
    ordered_goals = None
    rois = None
    list_deps = None
    desired_pos = None
    if stl_type_i==0:
        current_pos, desired_pos = random_free_cells(maze, 2)
        object = cell_to_object(env, desired_pos)
        final_stl = SimpleF(None, None, SimpleReach(0, object=object))
        curr_cells = [desired_pos]
        curr_objects = [object]
        stl_part_d[0] = final_stl
    elif stl_type_i==1:
        init_type = np.random.choice(["and", "or"], p=[0.5, 0.5])
        final_stl = SimpleListAnd([]) if init_type=="and" else SimpleListOr([])
        stack = [(0, init_type, final_stl)]
        curr_cells = []
        curr_objects = []
        free_pos_list = random_free_cells(maze, 16)
        current_pos = free_pos_list[0]   
        while len(stack)>0:
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
                    if next_node_type == "and":
                        next_stl = SimpleListAnd([])
                    else:
                        next_stl = SimpleListOr([])
                    stack.append([depth+1, next_node_type, next_stl])
                else:
                    next_node_type = "reach"
                    ta, tb = None, None
                    cell = free_pos_list[len(curr_objects)+1]
                    object = cell_to_object(env, cell)
                    reach_stl = SimpleF(ta, tb, SimpleReach(len(curr_objects), object=object))
                    stl_part_d[len(curr_objects)] = reach_stl
                    next_stl = reach_stl
                    curr_objects.append(object)
                    curr_cells.append(cell)
                curr_stl.children.append(next_stl)
        assignments = generate_assignments_stl(final_stl)
    elif stl_type_i==2:
        num_seqs = np.random.randint(2, 5)
        free_pos_list = random_free_cells(maze, num_seqs+1)
        current_pos = free_pos_list[0]
        desired_pos_list = free_pos_list[1:]
        desired_pos = free_pos_list[-1]
        poss_paths = plan_for_multi_goals(solver, current_pos, desired_pos_list)
        if len(poss_paths)>0:
            _, _, path_flat_list = poss_paths[0]
            ordered_goals = [path_list[-1] for path_list in path_flat_list]
            prev_stl = None
            for subgoal in ordered_goals[::-1]:
                cell = [subgoal[0], subgoal[1]]
                object = cell_to_object(env, cell)
                sub_stl = SimpleF(None, None, SimpleReach(len(curr_objects), object=object))
                if prev_stl is not None:
                    and_stl = SimpleAnd(left=sub_stl.children[0], right=prev_stl)
                    sub_stl.children = [and_stl]
                prev_stl = sub_stl
                stl_part_d[len(curr_objects)] = sub_stl
                curr_cells.append(cell)
                curr_objects.append(object)
            final_stl = sub_stl
        else:
            final_stl = None
    
    elif stl_type_i==3:
        t0 = time.time()
        on_path_key_likelihood = 0.4
        if args.nt<512:
            max_step = np.random.randint(4, 8)
        else:
            max_step = np.random.randint(4, 11)
        num_in_doors = min(np.random.randint(1, 3), max_step-2) # 2
        num_out_doors = np.random.randint(0, 4)
        
        my_maze = np.array(gen_hard_maze())
        current_pos = random_free_cell(my_maze)
        step, desired_pos, path = random_walk(my_maze, current_pos, max_step=max_step, visited={})
        door_path_indices = sorted(np.random.choice(range(1, max_step-1), num_in_doors, replace=False))
        door_cells = [path[idx] for idx in door_path_indices]
        indices = [0] + door_path_indices
        key_cells = []
        
        # add on-path doors
        my_maze_new = np.array(my_maze)
        for door_cell in door_cells:
            my_maze_new[door_cell[0], door_cell[1]] = 1
        visited = set(path)
        
        # add more doors off the path
        for i in range(num_out_doors):
            free_cell = random_free_cell(my_maze_new, visited)
            door_cells.append(free_cell)
            my_maze_new[free_cell[0], free_cell[1]] = 1
        
        # including init, goal, on-path doors, and out-path doors
        visited = set([desired_pos] + [current_pos] + door_cells)
        
        # now get keys
        feas = []
        for di, door_cell in enumerate(door_cells):
            # on-path doors, and there is a gap between two doors, add gap to the feasible slots to spawn keys
            if di < num_in_doors and door_path_indices[di] - indices[di] > 1:
                feas = feas + list(range(indices[di]+1, door_path_indices[di]))
            if len(feas)>0 and np.random.uniform(0, 1) < on_path_key_likelihood:
                key_idx = np.random.choice(feas)
                key_cells.append(path[key_idx])
                feas.remove(key_idx)
            else:
                for trial_i in range(100):
                    _, free_cell, _ = random_walk(my_maze_new, current_pos, max_step=np.random.randint(2, 4), visited=visited, type=1)
                    if free_cell is not None:
                        key_cells.append(free_cell)
                        break
                if free_cell is None:
                    free_cell = random_free_cell(my_maze_new, visited)
                    key_cells.append(free_cell)
                    
            for key_idx in feas:
                if path[key_idx]==key_cells[-1]:
                    # print("CONFLICT", key_idx, feas, path[key_idx], key_cells[-1])
                    feas.remove(key_idx)
                    break
                
            visited.add(key_cells[-1])
        
        rois = {key_cell:"k%d"%(k_i) for k_i, key_cell in enumerate(key_cells)}
        rois2 = {door_cell:"d%d"%(d_i) for d_i, door_cell in enumerate(door_cells)}
        rois.update(rois2)
        list_deps = [{"d%d"%ind:["k%d"%ind]} for ind in range(len(key_cells))]
        
        
        # gen stl
        curr_cells = []
        curr_objects = []
        key_to_obj_id = {}
        for loc in rois:
            key = rois[loc]
            assert key not in key_to_obj_id
            key_to_obj_id[key] = len(curr_objects)
            curr_cells.append(loc)
            curr_objects.append(cell_to_object(env, loc))
        
        required_stls = []
        for list_dep in list_deps:
            for door in list_dep:
                door_obj = curr_objects[key_to_obj_id[door]]
                depenencies = list_dep[door]
                for depend_on in depenencies:
                    depend_obj = curr_objects[key_to_obj_id[depend_on]]
                    until_stl = SimpleUntil(0, total_timesteps, 
                            left=SimpleNot(SimpleReach(key_to_obj_id[door], object=door_obj)), 
                            right=SimpleReach(key_to_obj_id[depend_on], object=depend_obj))
                    required_stls.append(until_stl)
                    
        goal_obj = cell_to_object(env, desired_pos)
        reach_goal_stl = SimpleF(None, None, SimpleReach(len(curr_objects), object=goal_obj))
        stl_part_d[len(curr_objects)] = reach_goal_stl
        curr_cells.append(desired_pos)
        curr_objects.append(goal_obj)
        required_stls.append(reach_goal_stl)
        final_stl = SimpleListAnd(required_stls)
    else:
        raise NotImplementedError
    
    return current_pos, desired_pos, curr_cells, curr_objects, final_stl, stl_part_d, assignments, poss_paths, ordered_goals, rois, list_deps

def update_stl_time_intervals(stl_type_i, tmp_cache, curr_objects, final_stl, stl_part_d, total_timesteps):
    if stl_type_i==0:
        intervals_d = get_subgoal_intervals_from_trajs(tmp_cache, curr_objects, stl_type_i=stl_type_i)
        update_f_oper_interval_with_g(final_stl, intervals_d[0])
    elif stl_type_i==1:
        intervals_d = get_subgoal_intervals_from_trajs(tmp_cache, curr_objects, mode="all", stl_type_i=stl_type_i)
        for obj_id in intervals_d:
            stl_part = stl_part_d[obj_id]
            if len(intervals_d[obj_id])==0: # we just assign 0, nt, with p=0.5 give G
                stl_part.ts = 0
                stl_part.te = total_timesteps
                if np.random.rand()>0.5: # add G
                    g_te = np.random.randint(1, total_timesteps//10)
                    g_stl = SimpleG(0, g_te, ap=stl_part.children[0])
                    stl_part.children[0] = g_stl
            else:
                inte = intervals_d[obj_id] 
                # add F times
                poss_times=[]
                for pair in inte:
                    poss_times.append(pair[0])
                    if pair[1] == total_timesteps and total_timesteps - pair[0]>10:
                        poss_times.append(pair[0] + np.random.randint(10, total_timesteps - pair[0]))
                    else:
                        poss_times.append(pair[1])
                margin = 5
                min_f_t = max(0, np.min(poss_times) - margin)
                max_f_t = min(np.max(poss_times) + margin, total_timesteps)
                stl_part.ts = min_f_t
                stl_part.te = max_f_t
                deltas = [xxx[1]-xxx[0] for xxx in inte]
                min_delta = np.min(deltas)
                if np.random.rand()>0.5 and min_delta>=5:
                    g_te = np.random.randint(1, min(min_delta-1, total_timesteps//10))
                    g_stl = SimpleG(0, g_te, ap=stl_part.children[0])
                    stl_part.children[0] = g_stl
    elif stl_type_i==2:
        intervals_d = get_subgoal_intervals_from_trajs(tmp_cache, curr_objects, mode="all", stl_type_i=stl_type_i)
        prev_stl_part_i = None
        for stage_i, stl_part_i in enumerate(list(stl_part_d.keys())[::-1]):
            stl_part = stl_part_d[stl_part_i]

            ts_list = [int(xxx[0]) for xxx in intervals_d[stl_part_i]]
            te_list = [int(xxx[1]) for xxx in intervals_d[stl_part_i]]
            if stage_i==0:
                stl_part.ts = 0
                stl_part.te = int(np.max(te_list))
            else:
                # delta_ts_list = [int(xxx[0])-int(yyy[0]) for xxx, yyy in zip(intervals_d[stl_part_i], intervals_d[prev_stl_part_i])]
                if len(intervals_d[stl_part_i])!=len(intervals_d[prev_stl_part_i]):
                    print("Error in intervals")
                    print(stl_part_i, prev_stl_part_i, intervals_d)
                    exit()
                delta_te_list = [int(xxx[1])-int(yyy[0]) for xxx, yyy in zip(intervals_d[stl_part_i], intervals_d[prev_stl_part_i])]
                stl_part.ts = 0
                stl_part.te = min(int(np.max(delta_te_list)) + 10, total_timesteps)
            
            delta = np.min([inte[1]-inte[0] for inte in intervals_d[stl_part_i]])
            if np.random.rand()>0.5 and delta >= 5: # add G
                g_te = np.random.randint(1, delta)
                if isinstance(stl_part.children[0], SimpleAnd):  # F (AND, (Reach, ...)) -> F (AND, (G(Reach), ...))
                    g_stl = SimpleG(0, g_te, ap=stl_part.children[0].children[0])
                    stl_part.children[0].children[0] = g_stl
                else:    # F (Reach, ...)) -> F (G(Reach), ...)
                    g_stl = SimpleG(0, g_te, ap=stl_part.children[0])
                    stl_part.children[0] = g_stl
            prev_stl_part_i = stl_part_i
    elif stl_type_i==3:
        intervals_d = get_subgoal_intervals_from_trajs(tmp_cache, [curr_objects[-1]], mode="min", stl_type_i=stl_type_i)
        update_f_oper_interval_with_g(final_stl.children[-1], intervals_d[0])
    else:
        raise NotImplementedError
    return

def rgbhex2dec(s):
    snew = s.replace("#", "")
    dec_list = [int(snew[0:2], 16), int(snew[2:4], 16), int(snew[4:6], 16)]
    return dec_list

def main():  
    # plan for a waypoint seq, then track control -> demo    
    total_timesteps = args.nt
    waypoint_threshold = args.waypoint_threshold
    task_sat_threshold = args.task_sat_threshold
    max_reach_step = args.max_reach_step
    num_mode_tries = args.num_mode_tries
    num_modes_per_stl = args.num_modes_per_stl
    num_ant_tries = args.num_ant_tries
    num_valid_demo_per_mode = args.num_valid_demo_per_mode
    utils.setup_exp_and_logger(args, test=args.test, dryrun=args.dryrun)
    
    # https://github.com/google-deepmind/dm_control/issues/207
    VIZ_DIR = args.viz_dir
    if utils.is_macos():
        os.environ['MUJOCO_GL'] = 'glfw'
    else:
        os.environ["MUJOCO_GL"] = "egl"
    
    maze = MockMaze(gen_hard_maze())
    solver = AStar(maze)
    if args.dynamics_type=="antmaze":
        env = gym.make("AntMaze_Large-v4", continuing_task=True, max_episode_steps=100_000, render_mode="rgb_array", use_contact_forces=False, maze_map=maze.maze_map, camera_id=-1)
        
        env.unwrapped.model.material('MatPlane').rgba[0] = 255 / 255 
        env.unwrapped.model.material('MatPlane').rgba[1] = 255 / 255 
        env.unwrapped.model.material('MatPlane').rgba[2] = 255 / 255
        
        env.unwrapped.model.geom('floor').rgba[0] = 0.1
        env.unwrapped.model.geom('floor').rgba[1] = 0.2
        env.unwrapped.model.geom('floor').rgba[2] = 0.3
        env.unwrapped.model.geom('floor').rgba[3] = 0.8
        env.unwrapped.model.vis.headlight.diffuse[:] = 0.1
        env.unwrapped.model.vis.headlight.specular[:] = 0.1
        
        mujoco_renderer = env.unwrapped.ant_env.mujoco_renderer
    elif args.dynamics_type=="pointmaze":
        env = gym.make("PointMaze_Large-v3", continuing_task=True, max_episode_steps=100_000, render_mode="rgb_array", maze_map=maze.maze_map, camera_id=-1)
        mujoco_renderer = env.unwrapped.point_env.mujoco_renderer
    else:
        raise NotImplementedError
    
    # TODO RENDERING CONFIG
    mujoco_renderer._get_viewer(render_mode="rgb_array")
    if utils.is_macos():
        mujoco_renderer.camera_id = -1         
        cam = mujoco_renderer._viewers["rgb_array"].cam
    else:
        cam = mujoco_renderer.viewer.cam
    cam.azimuth = 90
    cam.elevation = -90
    if args.dynamics_type=="antmaze":
        cam.distance = 60.0
    else:
        cam.distance = 15
    cam.lookat[:] = [0, 0, 0]  
    
    if args.dynamics_type=="antmaze":
        model = SAC.load(POLICY_FILE_PATH)
        def action_callback(obs, waypoint_xy):
            return model.predict(wrap_maze_obs(obs, waypoint_xy))[0]
        waypoint_controller = WaypointController(env.unwrapped.maze, action_callback)
    else:
        # https://minari.farama.org/tutorials/dataset_creation/point_maze_dataset/
        # {"p": 10.0, "d": -1.0}
        p_coeff = 10.0
        d_coeff = -1.0
        def action_callback(obs, waypoint_xy):
            action = (p_coeff * (waypoint_xy - obs["achieved_goal"]) + d_coeff * obs["observation"][2:])
            action = np.clip(action, -1, 1)
            return action
        waypoint_controller = WaypointController(env.unwrapped.maze, action_callback, 0.1)

    eta = utils.EtaEstimator(0, args.num_trials * 4)

    '''
    # can switch low-level controller {ball, ant}
    # for each possible stl syntax with object distribution
    #     for each modality (assignment, solver preference, etc)
    #         if path_solver can find len<X PWL path -> send to tracking controller
    #             for try_i in range(num_ant_tries)  # sometimes controller may crash
    #                 if ant reaches the goal point, calibrate intervals, and record this "stl, demo" pair
    '''
    n_errors = 0
    total_collected = 0
    total_stl_collected = 0
    data_list = []
    
    if args.dynamics_type=="antmaze":
        if args.nt==1024:
            max_reach_step=13
            max_reach_step1=15
            max_reach_step2=12
            max_reach_step3=None
        elif args.nt==512:
            max_reach_step=13
            max_reach_step1=15
            max_reach_step2=12
            max_reach_step3=None
        elif args.nt==384:
            max_reach_step=10
            max_reach_step1=10
            max_reach_step2=10
            max_reach_step3=12
        elif args.nt==256:
            max_reach_step=8
            max_reach_step1=10
            max_reach_step2=7
            max_reach_step3=10
    else:
        if args.nt==1024:
            max_reach_step=25
            max_reach_step1=28
            max_reach_step2=22
            max_reach_step3=20
        elif args.nt==512:
            max_reach_step=18
            max_reach_step1=20
            max_reach_step2=16
            max_reach_step3=20
        elif args.nt==384:
            max_reach_step=13
            max_reach_step1=14
            max_reach_step2=13
            max_reach_step3=16
        elif args.nt==256:
            max_reach_step=10
            max_reach_step1=12
            max_reach_step2=9
            max_reach_step3=12
    
    for seed_i in range(args.start_from, args.num_trials):
        for stl_type_i in [0,1,2,3]:
            seedseed = args.seed * args.num_trials * 4 + seed_i * 4 + stl_type_i
            utils.seed_everything(seedseed)
            eta.update()
            if seed_i%args.print_freq == 0 or seed_i==args.num_trials-1:
                print("Trial-%05d type:%d seed:%d | stls:%d trajs:%d ERRORs:%d | dt:%s elapsed:%s ETA:%s"%(
                    seed_i, stl_type_i, seedseed, total_stl_collected, total_collected, n_errors, eta.interval_str(), eta.elapsed_str(), eta.eta_str()))
            try:
                tmp_cache = []
                for spawn_trial_i in range(1000):
                    n_sat_modes = 0
                    ### random init obs/subgoal loc and stl syntax
                    current_pos, desired_pos, curr_cells, curr_objects, final_stl, stl_part_d, assignments, old_poss_paths, ordered_goals, rois, list_deps =\
                        rand_sample_loc_stl(stl_type_i, solver, env, maze, total_timesteps)
                    
                    if stl_type_i==2 and len(old_poss_paths)==0:
                        continue
                    
                    ### for different preference/modality
                    modalities = range(num_mode_tries) if stl_type_i!=1 else assignments
                    for mode_i, modal in enumerate(modalities):
                        n_sat_per_mode = 0
                        pwl_path_is_valid = False
                        
                        ### find a valid PWL
                        if stl_type_i==0: # just reach/stay with time
                            if 0<=mode_i<4:  # TODO consider different solver paths
                                bans = [list(solver.EXPLORATION_ACTIONS.values())[mode_i%4]]
                            else:
                                bans = []
                            fwdPath, path_flat = solver.generate_path(current_pos, desired_pos, with_flat=True, stoc_number=mode_i, banned_first_actions=bans)
                            pwl_path_is_valid = path_flat and len(path_flat) < max_reach_step
                            possd_list = [fwdPath]
                            path_flat_list = [path_flat]
                            goal_list=[desired_pos]
                        elif stl_type_i==1:  # and/or cases
                            desired_pos_list = [tuple(curr_cells[obj_id]) for obj_id in modal]
                            poss_paths = plan_for_multi_goals(solver, current_pos, desired_pos_list, max_path_len=max_reach_step1)
                            if len(poss_paths)>0:
                                pwl_path_is_valid = True
                                _, possd_list, path_flat_list = poss_paths[0]
                                goal_list=curr_cells
                        elif stl_type_i==2:
                            if 1<=mode_i<5:  # TODO consider different solver paths
                                bans = [list(solver.EXPLORATION_ACTIONS.values())[mode_i%4]]
                            else:
                                bans = []
                            if mode_i == 0:
                                poss_paths = old_poss_paths
                            else:
                                poss_paths = plan_for_multi_goals(solver, current_pos, ordered_goals, stoc_number=mode_i, banned_first_actions=bans, ordered=True, max_path_len=max_reach_step2)
                            if len(poss_paths)>0:
                                pwl_path_is_valid = True
                                _, possd_list, path_flat_list = poss_paths[0]
                                goal_list=ordered_goals
                        elif stl_type_i==3:
                            if 0<=mode_i<4:  # TODO consider different solver paths
                                bans = [list(solver.EXPLORATION_ACTIONS.values())[mode_i%4]]
                            else:
                                bans = []
                            res = solver.generate_path_with_deps(current_pos, desired_pos, None, bans, False, rois, list_deps, stoc_number=mode_i)
                            if res is not None and (max_reach_step3 is None or len(res[-1])<=max_reach_step3):
                                pwl_path_is_valid = True
                                possd_list, path_flat_list, planned_path =res
                                goal_list=[desired_pos]

                        ### valid PWL -> via tracking controler -> get ant trajs
                        if pwl_path_is_valid:
                            for ant_try_i in range(num_ant_tries):
                                record, is_sat = gen_trajs_from_waypoints(
                                    seed_i, stl_type_i, env, waypoint_controller, seedseed, 
                                    current_pos, possd_list, path_flat_list, total_timesteps, 
                                    waypoint_threshold, task_sat_threshold, goal_list=goal_list, rois=rois,
                                    fname_pattern="%s/ant_stl%d_seed%d_si%d_mo%d_tr%d_t%s.png"%(
                                        VIZ_DIR, stl_type_i, seedseed, seed_i, mode_i, ant_try_i, "%d"))
                                
                                # if this is okay
                                if is_sat:
                                    tmp_cache.append(record)
                                    n_sat_per_mode += 1
                                    if n_sat_per_mode == (num_valid_demo_per_mode *3 if stl_type_i==1 else num_valid_demo_per_mode):
                                        break
                        
                        if n_sat_per_mode>0:
                            n_sat_modes += 1
                            if n_sat_modes == num_modes_per_stl:
                                break
                    
                    if n_sat_modes>0:  # now we need to calibrate the time intervals for the F                    
                        update_stl_time_intervals(stl_type_i, tmp_cache, curr_objects, final_stl, stl_part_d, total_timesteps)
                        collect_stl_records(data_list, tmp_cache, final_stl, curr_objects)
                        total_stl_collected += 1
                        total_collected += len(tmp_cache)
                        break
            except KeyboardInterrupt:
                raise  # Re-raise KeyboardInterrupt
            except Exception as e:
                # Handle all other exceptions
                print("An exception occurred at seed_i:%d stl_type_i:%d spawn_trial_i:%d mode_i:%d ant_try_i:%d, seedseed:%d"%(
                    seed_i, stl_type_i, spawn_trial_i, mode_i, ant_try_i, seedseed))
                print(f"Error: {e}")
                print(traceback.format_exc())
                n_errors+=1
                # raise
        
        # intervally save the data
        if seed_i % 100 == 0 or seed_i == args.num_trials-1:
            np.savez("%s/data.npz"%(args.exp_dir_full), data=data_list)  
        if seed_i+1 in [100, 1000, 10000]:
            np.savez("%s/data_%d.npz"%(args.exp_dir_full, seed_i+1), data=data_list)
    return


'''
https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/main/scripts/D4RL/antmaze/create_antmaze_dataset.py
https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/main/scripts/D4RL/antmaze/controller.py
https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/main/scripts/pointmaze/maze_solver.py
https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/main/scripts/D4RL/antmaze/reach_goal_ant.py
https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/main/scripts/D4RL/antmaze/train_ant.py
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--seed", type=int, default=1007)
    add("--exp_name", '-e', type=str, default="data_ant_DBG")
    add("--gpus", type=str, default="0")
    add("--cpu", action='store_true', default=False)
    add("--test", action='store_true', default=False)
    add("--dryrun", action='store_true', default=False)
    add("--num_trials", type=int, default=100)
    add("--nt", type=int, default=1000)
    add("--no_viz", action='store_true', default=False)
    
    add("--viz_freq", type=int, default=50)
    add("--print_freq", type=int, default=10)
    
    add("--waypoint_threshold", type=float, default=0.5)
    add("--task_sat_threshold", type=float, default=0.35)
    add("--max_reach_step", type=int, default=13)
    add("--num_mode_tries", type=int, default=10)
    add("--num_modes_per_stl", type=int, default=4)
    add("--num_ant_tries", type=int, default=5)
    add("--num_valid_demo_per_mode", type=int, default=1)
    add("--start_from", type=int, default=0)
    add("--dynamics_type", type=str, choices=['antmaze', 'pointmaze'], default="antmaze")
    args = parser.parse_args()
    
    assert args.nt in [128, 256, 384, 512, 1024]
    
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda:0"
    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.3f seconds"%(t2-t1))