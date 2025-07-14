import os
from os.path import join as ospj
import sys
import time
import math
import shutil
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import imageio
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle

############################### 
THE_EXP_ROOT_DIR="exps_gstl"  # 
###############################

import torch

def is_macos():
    return sys.platform == "darwin"

def create_custom_lr_scheduler(optimizer, warmup_epochs=0, warmup_lr=None, decay_epochs=0, decay_lr=None, decay_mode="cosine"):
    lr = optimizer.param_groups[0]['lr']
    if decay_epochs != 0:
        final_lr = decay_lr
    else:
        final_lr = lr
    
    def customize_func(epoch):
        if warmup_epochs != 0 and epoch < warmup_epochs:
            start_ratio = warmup_lr / lr
            end_ratio = 1
            #print("WARMUP", start_ratio, end_ratio)
            return (end_ratio - start_ratio) / (warmup_epochs-1) * epoch + start_ratio
        elif decay_epochs != 0 and epoch - warmup_epochs < decay_epochs:
            if decay_mode=="cosine":
                eta_min = decay_lr / lr
                eta_max = 1.0
                T_curr = epoch - warmup_epochs
                T_max = decay_epochs
                eta_t = eta_min + 1/2*(eta_max-eta_min)*(1+np.cos((T_curr/T_max)*np.pi))
                return eta_t
            else:
                start_ratio = 1.0
                end_ratio = decay_lr / lr
                return (end_ratio - start_ratio) / (decay_epochs-1) * (epoch - warmup_epochs) + start_ratio
        else:
            #print("FT", final_lr)
            return final_lr/lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=customize_func)    
    return scheduler

def create_custom_lr_scheduler_bak(optimizer, total_epochs, lr, warmup_epochs=0, warmup_lr=None, decay_epochs=0, decay_lr=None, decay_mode="cosine", verbose=False):
    left_epochs = total_epochs
    constant_lr = lr
    scheduler_list=[]
    if warmup_epochs != 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_lr/lr, total_iters=warmup_epochs, verbose=verbose)
        left_epochs -= warmup_epochs
        scheduler_list.append(warmup_scheduler)
    
    if decay_epochs != 0:
        if decay_mode=="cosine":
            decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=decay_lr, verbose=verbose)
        else:
            decay_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=decay_lr/lr, total_iters=decay_epochs, verbose=verbose)
        left_epochs -= decay_epochs
        constant_lr = decay_lr
        scheduler_list.append(decay_scheduler)
    
    constant_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=constant_lr / lr, total_iters=left_epochs, verbose=verbose)
    scheduler_list.append(constant_scheduler)
    
    return torch.optim.lr_scheduler.ChainedScheduler(scheduler_list)

class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs, lr, warmup_epochs=0, warmup_lr=None, decay_epochs=0, decay_lr=None, decay_mode="cosine"):
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.left_epochs = total_epochs
        
        constant_lr = lr
        
        if warmup_epochs != 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_lr/lr, total_iters=warmup_epochs, verbose=True)
            self.left_epochs -= warmup_epochs
        
        if decay_epochs != 0:
            if decay_mode=="cosine":
                self.decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=decay_lr, verbose=True)
            else:
                self.decay_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=decay_lr/lr, total_iters=decay_epochs)
            self.left_epochs -= decay_epochs
            constant_lr = decay_lr
        
        self.constant_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=constant_lr / lr, total_iters=self.left_epochs, verbose=True)
        
        super().__init__(optimizer)
    
    def step(self, epoch=None):
        if self.warmup_epochs!=0 and self._step_count-1 < self.warmup_epochs:
            print(self._step_count-1, "WARM_UP")
            self.warmup_scheduler.step()
        elif self.decay_epochs!=0 and self._step_count-1 - self.warmup_epochs < self.decay_epochs:
            print(self._step_count-1, "DECAY")
            self.decay_scheduler.step()
        else:
            print(self._step_count-1)
            self.constant_scheduler.step()
        self._step_count += 1
    

def save_model_freq_last(state_dict, model_dir, epi, save_freq, epochs, ema=False):
    key_epochs = [10, 50, 100, 200, 300, 500, 800, 1000, 1200, 1500, 2000]
    if epi!=0 and (epi % save_freq == 0 or epi == epochs-1 or epi in key_epochs):
        torch.save(state_dict, "%s/model_%05d.ckpt"%(model_dir, epi))
    if epi % 10 == 0 or epi == epochs-1:
        torch.save(state_dict, "%s/model_last.ckpt"%(model_dir))

def plt_save_close(img_path, bbox_inches='tight', pad_inches=0.1):
    plt.savefig(img_path, bbox_inches=bbox_inches, pad_inches=pad_inches)
    plt.close()

def get_exp_dir(just_local=False):
    if just_local:
        return "./"
    else:
        if "MY_EXPS_DIR" not in os.environ:
            # exit("CANNOT FIND ENV VARIABLE for 'MY_EXPS_DIR':%s"%(dataroot))
            dataroot="../../"
        else:
            dataroot=os.environ["MY_EXPS_DIR"]
        dataroot=os.path.join(dataroot, THE_EXP_ROOT_DIR)
        return dataroot


def get_model_path(pretrained_path, just_local=False):
    return ospj(get_exp_dir(just_local), smart_path(pretrained_path))

def to_np(x):
    return x.detach().cpu().numpy()

def to_torch(x):
    return torch.from_numpy(x).float().cuda()

def uniform_tensor(amin, amax, size):
    return torch.rand(size) * (amax - amin) + amin

def rand_choice_tensor(choices, size):
    return torch.from_numpy(np.random.choice(choices, size)).float()

def to_np_dict(di):
    di_np = {}
    for key in di:
        di_np[key] = to_np(di[key])
    return di_np

def dict_to_cuda(batch):
    cuda_batch = {}
    for key in batch:
        cuda_batch[key] = batch[key]
        if hasattr(batch[key], "device"):
            cuda_batch[key] = cuda_batch[key].cuda()
    return cuda_batch

def dict_to_torch(batch, keep_keys=[]):
    torch_batch = {}
    for key in batch:
        if key in keep_keys:
            torch_batch[key] = batch[key]
        else:
            torch_batch[key] = torch.from_numpy(batch[key])
    return torch_batch


def build_relu_nn(input_dim, output_dim, hiddens, activation_fn=torch.nn.ReLU, last_fn=None):
    n_neurons = [input_dim] + hiddens + [output_dim]
    layers = []
    for i in range(len(n_neurons)-1):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
        layers.append(activation_fn())
    if last_fn is not None:
        layers[-1] = last_fn()
    else:
        del layers[-1]
    return nn.Sequential(*layers)


def build_relu_nn1(input_output_dim, hiddens, activation_fn=torch.nn.ReLU, last_fn=None):
    return build_relu_nn(input_output_dim[0], input_output_dim[1], hiddens, activation_fn, last_fn=last_fn)

def generate_gif(gif_path, duration, fs_list):
    with imageio.get_writer(gif_path, mode='I', duration=duration, loop=0) as writer:
        for filename in fs_list:
            image = imageio.imread(filename)
            writer.append_data(image)

def soft_step(x):
    return (torch.tanh(500 * x) + 1)/2

def soft_step_hard(x):
    hard = (x>=0).float()
    soft = (torch.tanh(500 * x) + 1)/2
    return soft + (hard - soft).detach()

def xxyy_2_Ab(x_input):
    xmin, xmax, ymin, ymax = x_input
    A = np.array([
            [-1, 1, 0, 0],
            [0, 0, -1, 1]
        ]).T
    b = np.array([-xmin, xmax, -ymin, ymax])
    return A, b

def xyr_2_Ab(x, y, r, num_edges=8):
    thetas = np.linspace(0, np.pi*2, num_edges+1)[:-1]
    A = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)
    b = r + x * np.cos(thetas) + y * np.sin(thetas)
    return A, b

def find_path(path, just_local=False):
    return os.path.join(get_exp_dir(just_local), path)

def find_npz_path(path, just_local=False):
    if ".npz" not in path:
        path = os.path.join(path, "cache.npz")
    if path.startswith("/"):
        path = path
    else:
        path = os.path.join(get_exp_dir(just_local=just_local), path)
    return path

def smart_path(s):
    if ".ckpt" not in s:
        s = s+"/models/model_last.ckpt"
    return s

class MyTimer():
    def __init__(self):
        self.timestamp = {}
        self.count = {}
        self.profile = {}
        self.left = {}
        self.right = {}
        self.last = None
    
    def add(self, key, new_name=None):
        self.timestamp[key] = time.time()
        if key not in self.count:
            self.count[key] = 0 
        self.count[key] += 1
        
        if self.last is not None and self.count[key]==self.count[self.last]:
            if new_name is None:
                new_name = "%s-%s"%(key, self.last)
            self.left[new_name] = key
            self.right[new_name] = self.last
            dt = self.timestamp[key] - self.timestamp[self.last]
            if new_name not in self.profile:
                self.profile[new_name] = 0
            self.profile[new_name] += dt
        
        self.last = key
    
    def print_profile(self):
        s=""
        for key in self.profile:
            left = self.left[key]
            right = self.right[key]
            tsum = self.profile[key]
            cnt = self.count[left]
            s += "%s:%.3f "%(key, tsum/ cnt)
        print(s)


class EtaEstimator():
    def __init__(self, start_iter, end_iter, check_freq=1, epochs=None, total_train_bs=None, total_val_bs=None, batch_size=None, viz_freq=None, num_workers=1):
        self.start_iter = start_iter
        num_workers = 1# if num_workers is None else num_workers
        self.end_iter = end_iter//num_workers
        self.check_freq = check_freq
        self.curr_iter = start_iter
        self.start_timer = None
        self.interval = 0
        self.eta_t = 0
        self.num_workers = num_workers

        self.viz_freq = viz_freq
        self.epochs = epochs
        self.batch_size = batch_size
        self.prev_is_viz = False
        self.prev_timer = time.time()
        self.nn_stat_train = []
        self.nn_train_bs = []
        self.nn_stat_val = []
        self.nn_val_bs = []
        self.viz_stat = []
        self.total_train_bs = total_train_bs
        self.total_val_bs = total_val_bs

    def update(self):
        if self.start_timer is None:
            self.start_timer = time.time()
        self.curr_iter += 1
        # if self.curr_iter % (max(1,self.check_freq//self.num_workers)) == 0:
        self.interval = self.elapsed() / (self.curr_iter - self.start_iter)        
        self.eta_t = self.interval * (self.end_iter - self.curr_iter)
    
    def update_viz_time(self, duration):
        self.viz_stat.append(duration)

    def smart_update(self, epi, duration=None, bs=None, mode=None, bi=None, is_viz=False):
        self.curr_epi = epi
        self.curr_stage = mode
        self.curr_bi = bi
        nn_time = duration
        if mode=="train":
            self.nn_stat_train.append(nn_time)
            self.nn_train_bs.append(bs)
        elif mode=="val":
            self.nn_stat_val.append(nn_time)
            self.nn_val_bs.append(bs)
        else:
            raise NotImplementedError
        
        if len(self.nn_stat_train)>0:
            if len(self.nn_stat_train)>1:
                nn_train_sum = np.sum(self.nn_stat_train[1:])
                nn_train_bs = np.sum(self.nn_train_bs[1:])
            else:
                nn_train_sum = np.sum(self.nn_stat_train)
                nn_train_bs = np.sum(self.nn_train_bs)
            nn_train_avg_per_sample = nn_train_sum / nn_train_bs
        if len(self.nn_stat_val)>0:
            nn_val_sum = np.sum(self.nn_stat_val)
            nn_val_bs = np.sum(self.nn_val_bs)
            nn_val_avg_per_sample = nn_val_sum / nn_val_bs
        else:
            nn_val_avg_per_sample = nn_train_avg_per_sample
        # print("t_avg_per_sample", nn_train_avg_per_sample, nn_val_avg_per_sample)
        # print("STAT", self.total_train_bs, self.total_val_bs, self.curr_bi)
        remain_epis = self.epochs - self.curr_epi - 1
        if self.curr_stage == "train":
            remain_train_bs = self.total_train_bs - (self.curr_bi+1) * self.batch_size
            remain_val_bs = self.total_val_bs
             
        elif self.curr_stage == "val":
            remain_train_bs = 0
            remain_val_bs = self.total_val_bs - (self.curr_bi+1) * self.batch_size

        remain_single_time = remain_train_bs * nn_train_avg_per_sample+ remain_val_bs * nn_val_avg_per_sample
       
        if len(self.viz_stat)>0:
            avg_viz_time = np.mean(self.viz_stat)
        else:
            avg_viz_time = 1 * nn_train_avg_per_sample

        viz_cnt=0
        # viz_cnt = (self.epochs - self.curr_epi - 1)
        for ii in range(self.curr_epi, self.epochs):
            if ii % self.viz_freq == 0 or ii == self.epochs-1:
                viz_cnt += 1

        remain_viz_time = viz_cnt * avg_viz_time
        # print("CAL", remain_epis * (nn_train_avg_per_sample*self.total_train_bs + nn_val_avg_per_sample*self.total_val_bs), remain_single_time, remain_viz_time)

        self.eta_t_smart = remain_epis * (nn_train_avg_per_sample*self.total_train_bs + nn_val_avg_per_sample*self.total_val_bs) +\
                           remain_single_time + remain_viz_time

        if is_viz==False:
            self.update()
                
        self.prev_is_viz = is_viz
        self.prev_timer = time.time()

    def elapsed(self):
        return time.time() - self.start_timer
    
    def eta(self):
        return self.eta_t
    
    def elapsed_str(self):
        return time_format(self.elapsed())
    
    def interval_str(self):
        return time_format(self.interval)

    def eta_str(self):
        return time_format(self.eta_t)

    def eta_str_smart(self):
        return time_format(self.eta_t_smart)

def time_format(secs):
    _s = secs % 60 
    _m = secs % 3600 // 60
    _h = secs % 86400 // 3600
    _d = secs // 86400
    if _d != 0:
        return "%02dD%02dh%02dm%02ds"%(_d, _h, _m, _s)
    else:
        if _h != 0:
            return "%02dH%02dm%02ds"%(_h, _m, _s)
        else:
            if _m != 0:
                return "%02dm%02ds"%(_m, _s)
            else:
                return "%05.2fs"%(_s)


def uniform(a, b, size):
    return torch.rand(*size) * (b - a) + a

def linspace(a, b, size):
    return torch.from_numpy(np.linspace(a, b, size)).float()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# TODO create the exp directory
def setup_exp_and_logger(args, set_gpus=True, just_local=False, test=False, dryrun=False):
    seed_everything(args.seed)
    if set_gpus and hasattr(args, "gpus") and args.gpus is not None:
        os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # Solve core dump problems
    # Get all available CPUs and remove 8,9
    all_cpus = set(range(os.cpu_count()))  # Get all CPU numbers
    if len(all_cpus)<36 and is_macos()==False:
        bad_cpus = {8, 9}  # The ones we want to avoid
        allowed_cpus = all_cpus - bad_cpus  # Remove the bad ones
        os.sched_setaffinity(0, allowed_cpus)
        print("Process is now restricted to CPUs:", os.sched_getaffinity(0))
    
    if dryrun:
        return
    sys.stdout = logger = Logger()
    EXP_ROOT_DIR = get_exp_dir(just_local)
    if test:
        if hasattr(args, "rl") and args.rl:
            tuples = args.rl_path.split("/")
        else:
            tuples = args.net_pretrained_path.split("/")
        if ".ckpt" in tuples[-1] or ".zip" in tuples[-1] :
            EXP_ROOT_DIR = ospj(EXP_ROOT_DIR, tuples[-3])
        else:
            EXP_ROOT_DIR = ospj(EXP_ROOT_DIR, tuples[-1])
        if hasattr(args, "fix") and args.fix is not None:
            if args.suffix is not None:
                args.suffix = ("_"+args.suffix).replace("__", "_")
            else:
                args.suffix = ""
            if args.seed != 1007:
                seed_str = "_%d"%(args.seed)
            else:
                seed_str = ""
            args.exp_dir_full = os.path.join(EXP_ROOT_DIR, "results_test%s%s"%(args.suffix, seed_str))
        else:
            if hasattr(args, "suffix") and args.suffix is not None:
                suffix="_"+args.suffix
            else:
                suffix=""
            args.exp_dir_full = os.path.join(EXP_ROOT_DIR, "test_%s%s" % (logger._timestr, suffix))
    elif hasattr(args, "batch") and args.batch is not None:
        args.exp_dir_full = os.path.join(EXP_ROOT_DIR, "g%s_BATCH_%s" % (logger._timestr, args.exp_name), "job_%04d" % (args.job_id))
    elif args.exp_name.startswith("batch"):
        args.exp_dir_full = os.path.join(EXP_ROOT_DIR, "g%s_SEED_%s" % (logger._timestr, args.exp_name), "seed_%d" % (args.seed))
    else:
        keys = ["exp"]
        if any([args.exp_name.startswith(key) for key in keys]) and "debug" not in str.lower(args.exp_name) and "dbg" not in str.lower(args.exp_name):
            args.exp_dir_full = os.path.join(EXP_ROOT_DIR, args.exp_name)
        else:
            args.exp_dir_full = os.path.join(EXP_ROOT_DIR, "g%s_%s" % (logger._timestr, args.exp_name))
    args.viz_dir = os.path.join(args.exp_dir_full, "viz")
    args.src_dir = os.path.join(args.exp_dir_full, "src")
    args.model_dir = os.path.join(args.exp_dir_full, "models")
    os.makedirs(args.viz_dir, exist_ok=True)
    os.makedirs(args.src_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    for fname in os.listdir('./'):
        if fname.endswith('.py'):
            shutil.copy(fname, os.path.join(args.src_dir, fname))

    logger.create_log(args.exp_dir_full)
    write_cmd_to_file(args.exp_dir_full, sys.argv)
    np.savez(os.path.join(args.exp_dir_full, 'args'), args=args)

    return args



# TODO logger
class Logger(object):
    def __init__(self):
        self._terminal = sys.stdout
        self._timestr = datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")

    def create_log(self, log_path):
        self.log = open(log_path + "/log-%s.txt" % self._timestr, "a", 1)

    def write(self, message):
        self._terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def write_cmd_to_file(log_dir, argv):
    with open(ospj(log_dir, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(argv))


# TODO metrics
class MeterDict:
    def __init__(self):
        self.d = {}
    
    def reset(self):
        del self.d
        self.d = {}

    def update(self, key, val):
        if key not in self.d:
            # curr, count, avg
            self.d[key] = [val, 1, val]
        else:
            _, count, avg = self.d[key]
            self.d[key][0] = val
            self.d[key][1] = count+1
            ratio = 1 / (count+1)
            self.d[key][2] = avg * (1-ratio) + val * ratio
    
    def get_val(self, key):
        return self.d[key][0]

    def __getitem__(self, key):
        return self.get_val(key)

    def get_avg(self, key):
        return self.d[key][2]
    
    def __contains__(self, key):
        return key in self.d

    def __call__(self, key):
        return self.get_avg(key)

def get_n_meters(n):
    return [AverageMeter() for _ in range(n)]

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.history.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# geometry checking
def cross_product(x1, y1, x2, y2):
    return x1 * y2 - x2 * y1


def inner_product(x1, y1, x2, y2):
    return x1 * x2 + y1 * y2


def generate_bbox(x, y, theta, L, W):
    # (2, 5)
    bbox=np.array([
        [L/2, W/2],
        [L/2, -W/2],
        [-L/2, -W/2],
        [-L/2, W/2],
    ]).T

    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])

    # (2, 1)
    trans = np.array([[
        x, y
    ]]).T
    new_bbox = (rot @ bbox) + trans
    return new_bbox.T


def get_anchor_point(x, y, th, L, W, num_L, num_W):
    x1 = L/2
    y1 = W/2
    x2 = -L/2
    y2 = W/2
    x3 = -x1
    y3 = -y1
    x4 = -x2
    y4 = -y2
    r_l = L / num_L / 2
    r_w = W / num_W / 2
    r = torch.minimum(torch.maximum(r_l, r_w), W / 2)

    poly = torch.stack([x1, y1, x2, y2, x3, y3, x4, y4], dim=-1).reshape(list(x1.shape) + [4, 2])
    poly_x = poly[..., 0] * torch.cos(th[..., None]) - poly[..., 1] * torch.sin(th[..., None]) + x[..., None]
    poly_y = poly[..., 0] * torch.sin(th[..., None]) + poly[..., 1] * torch.cos(th[..., None]) + y[..., None]
    poly = torch.stack([poly_x, poly_y], dim=-1)

    alpha = torch.linspace(0, 1, num_L).to(x1.device)
    beta = torch.linspace(0, 1, num_W).to(x1.device)
    xs_ = (x2 + r)[..., None] * (1 - alpha) + (x1 - r)[..., None] * alpha # (N, T, k1)
    ys_ = (y3 + r)[..., None] * (1 - beta) + (y2 - r)[..., None] * beta # (N, T, k2)
    # xys_ = torch.stack(torch.meshgrid(xs_, ys_), dim=-1).reshape(list(xs.shape[:-1]) + [num_L*num_W, 2])

    batch_size = list(x1.shape)
    xs_ = xs_[..., None].expand(batch_size+ [num_L, num_W]).reshape(batch_size +[num_L*num_W])
    ys_ = ys_[..., None, :].expand(batch_size+ [num_L, num_W]).reshape(batch_size +[num_L*num_W])
    # print(xs_.shape, ys_.shape, th.shape, th[..., None].shape)
    xs = xs_ * torch.cos(th[..., None]) - ys_ * torch.sin(th[..., None]) + x[..., None]
    ys = xs_ * torch.sin(th[..., None]) + ys_ * torch.cos(th[..., None]) + y[..., None]
    # xys = torch.stack(torch.meshgrid(xs, ys), dim=-1).reshape(list(xs.shape[:-1]) + [num_L*num_W, 2])  # (N, T, k1*k2, 2)
    xys = torch.stack([xs, ys], dim=-1)
    return poly, xys, r

WALL = 10
EMPTY = 11
GOAL = 12
        
def parse_maze(maze_str):
    lines = maze_str.strip().split('\\')
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == '#':
                maze_arr[w][h] = WALL
            elif tile == 'G':
                maze_arr[w][h] = GOAL
            elif tile == ' ' or tile == 'O' or tile == '0':
                maze_arr[w][h] = EMPTY
            else:
                raise ValueError('Unknown tile type: %s' % tile)
    return maze_arr

def get_maze():
    return parse_maze(MEDIUM_MAZE)


def plot_maze(maze0):
    maze = maze0.T
    size = 1.0
    # offset_x = -0.5 # 0.0
    # offset_y = -0.5 # 0.0
    offset_x = -.75
    offset_y = -.75
    max_x = offset_x + size * maze.shape[1]
    max_y = offset_y + size * maze.shape[0]
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            lowerleft_x = offset_x + size * j
            # lowerleft_y = offset_y + size * (maze.shape[0]-1) - size * i
            lowerleft_y = offset_y + size * i
            if maze[i, j] == EMPTY:
                color = "white"
            elif maze[i, j] == WALL:
                color = "black"
            rect = Rectangle([lowerleft_x, lowerleft_y], size, size, edgecolor="white", facecolor=color, alpha=0.8)
            ax = plt.gca()
            ax.add_patch(rect)
    plt.axis("scaled")
    plt.xlim(offset_x-0.25, max_x + 0.25)
    plt.ylim(offset_y-0.25, max_y + 0.25)
    info = {"xmin": offset_x-0.25, "xmax": max_x + 0.25, "ymin":offset_y-0.25, "ymax":max_y + 0.25, }
    return info

LARGE_MAZE = \
        "############\\"+\
        "#OOOO#OOOOO#\\"+\
        "#O##O#O#O#O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O####O###O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "##O#O#O#O###\\"+\
        "#OO#OOO#OGO#\\"+\
        "############"

LARGE_MAZE_EVAL = \
        "############\\"+\
        "#OO#OOO#OGO#\\"+\
        "##O###O#O#O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "#O##O#OO##O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O##O#O#O###\\"+\
        "#OOOO#OOOOO#\\"+\
        "############"

MEDIUM_MAZE = \
        '########\\'+\
        '#OO##OO#\\'+\
        '#OO#OOO#\\'+\
        '##OOO###\\'+\
        '#OO#OOO#\\'+\
        '#O#OO#O#\\'+\
        '#OOO#OG#\\'+\
        "########"

MEDIUM_MAZE_EVAL = \
        '########\\'+\
        '#OOOOOG#\\'+\
        '#O#O##O#\\'+\
        '#OOOO#O#\\'+\
        '###OO###\\'+\
        '#OOOOOO#\\'+\
        '#OO##OO#\\'+\
        "########"

SMALL_MAZE = \
        "######\\"+\
        "#OOOO#\\"+\
        "#O##O#\\"+\
        "#OOOO#\\"+\
        "######"

U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

U_MAZE_EVAL = \
        "#####\\"+\
        "#OOG#\\"+\
        "#O###\\"+\
        "#OOO#\\"+\
        "#####"

OPEN = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OOGOO#\\"+\
        "#OOOOO#\\"+\
        "#######"