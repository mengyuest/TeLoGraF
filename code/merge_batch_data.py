import numpy as np
import os
import time
import argparse
import utils
import torch
from os.path import join as ospj
from generate_scene_v1 import generate_trajectories_dubins
from generate_scene_v1 import find_ap_in_lines

def get_stat(x):
    return np.min(x, axis=0), np.max(x, axis=0), np.mean(x, axis=0), np.std(x, axis=0)

def get_nt_flatten(x):
    shape = x.shape # assume first dim N, and second dim T
    return x.reshape(shape[0]*shape[1], *shape[2:])

def main():
    EXP_DIR = utils.get_exp_dir()
    if args.env=="simple":
        # dict_keys(['ego', 'stl', 'objects', 'goals_indices', 'obstacles_indices', 'stl_seed', 'stl_type_i', 'trial_i', 'score', 'us', 'state'])
        # input_data_paths=[
        #     "g0102-075001_BATCH_data4x500x20",
        #     "g0102-075002_BATCH_data4x500x20"] + \
        input_data_paths=\
            [xxx.strip() for xxx in 
            '''g0124-055040_BATCH_data4x500x20_v0
            g0124-113606_BATCH_data4x500x20_v1
            g0123-180220_BATCH_data1x2000x5_s3_1
            g0123-182308_BATCH_data1x500x20_s3
            g0123-182329_BATCH_data1x1000x4_s3_0
            g0123-182336_BATCH_data1x1000x4_s3_1
            g0123-182342_BATCH_data1x1000x4_s3_2
            g0123-182352_BATCH_data1x1000x4_s3_3
            g0123-204457_BATCH_data1x1000x4_s3_5
            g0123-204502_BATCH_data1x1000x4_s3_6
            g0123-204507_BATCH_data1x1000x4_s3_7
            g0123-210857_BATCH_data1x1000x4_s3_8'''.split("\n")]
        out_data_path = "data_0_simple_v0"
    elif args.env=="dubins":
        input_data_paths=[
            'g0116-033319_BATCH_dubins4x500x10',
            'g0124-154543_BATCH_dubins4x1200x4',
            'g0124-154605_BATCH_dubins4x1200x4',
            'g0124-154606_BATCH_dubins4x1200x4',
            'g0124-154608_BATCH_dubins4x1200x4',
            'g0124-161641_BATCH_dubins4x500x4_0',
            'g0124-161919_BATCH_dubins4x500x4_1',
            'g0124-161926_BATCH_dubins4x500x4_2',
            'g0124-161940_BATCH_dubins4x500x4_3',
            'g0124-162708_BATCH_dubins4x500x4_4',
            'g0124-162923_BATCH_dubins4x500x4_5',
            'g0124-162924_BATCH_dubins4x500x4_6',
            'g0124-162925_BATCH_dubins4x500x4_7',
            'g0124-164640_BATCH_dubins4x500x4_4',
            'g0124-164642_BATCH_dubins4x500x4_5',
            'g0124-164645_BATCH_dubins4x500x4_6',
            'g0124-164648_BATCH_dubins4x500x4_7',
            'g0124-170438_BATCH_dubins4x500x4_4',
            'g0124-170545_BATCH_dubins4x500x4_5',
            'g0124-171610_BATCH_dubins4x500x4_4',
            'g0124-171617_BATCH_dubins4x500x4_6',
            'g0124-171625_BATCH_dubins4x500x4_7',
        ]
        out_data_path = "data_1_dubins_v0"
    elif args.env=="pointmaze":
        input_data_paths=[
            "g0127-022542_point_data_10k_v3",
            "g0127-032902_point_data_10k_v4",
        ]
        out_data_path = "data_2_pointmaze_v0"
    elif args.env=="antmaze":
        input_data_paths=[
            "g0114-002204_ant_data_2k5_512",
            "g0127-082711_ant_data_2k5_512_new2008",
            "g0127-083500_ant_data_2k5_512_new2009",
            "g0127-090136_ant_data_2k5_512_new2010",
            # "g0127-090317_ant_data_2k5_512_new2011"
        ]
        out_data_path = "data_3_antmaze_v0"
    elif args.env=="panda":
        input_data_paths=[
        "g0119-165100_panda_1007_s0",
        "g0119-165111_panda_1007_s1",
        "g0119-165117_panda_1007_s2",
        "g0119-165122_panda_1007_s3",
        "g0119-021055_panda_1008",
        "g0119-160702_panda_1009_s0",
        "g0119-160745_panda_1009_s1",
        "g0119-160755_panda_1009_s2",
        "g0119-110855_panda_1009_s3",
        "g0119-183720_panda_1010_s0",
        "g0119-183728_panda_1010_s1",
        "g0119-183736_panda_1010_s2",
        "g0119-183742_panda_1010_s3",
        "g0119-191514_panda_1011_s0",
        "g0119-191524_panda_1011_s1",
        "g0119-191531_panda_1011_s2",
        "g0119-191537_panda_1011_s3",
        "g0119-070746_panda_1012",
        "g0119-070755_panda_1013",
        "g0119-073049_panda_1014",
        "g0119-074401_panda_1015",
        "g0119-074402_panda_1016",
        "g0119-022246_panda_1017",
        "g0119-074525_panda_1018",
        
        "g0124-194746_panda200_2024_s0",
        "g0124-212429_panda200_2025_s0",
        "g0125-014530_panda4x500_0",
        "g0125-022818_panda4x500_0",
        "g0125-024940_panda4x500_0",
        "g0125-025449_panda4x500_0",
        "g0125-032050_panda4x500_0",
        "g0125-032658_panda4x500_0",
        "g0125-040120_panda4x500_0",
        "g0125-043545_panda4x500_0",
        "g0126-071321_panda4x500_40",
        "g0126-071321_panda4x500_41",
        "g0126-071537_panda4x500_42",
        "g0126-071537_panda4x500_43",
        "g0126-071603_panda4x500_44",
        "g0126-071603_panda4x500_45",
        "g0126-071648_panda4x500_46",
        "g0126-071648_panda4x500_47",
        "g0126-071722_panda4x500_48",
        "g0126-071722_panda4x500_49",
        "g0126-071750_panda4x500_50",
        "g0126-071750_panda4x500_51",
        "g0126-072049_panda4x500_52",
        "g0126-072049_panda4x500_53",
        "g0126-072319_panda4x500_54",
        "g0126-072319_panda4x500_55",
    ]
        out_data_path = "data_4_panda_v0"
    else:
        raise NotImplementedError 
    
    output_full_path = ospj(EXP_DIR, out_data_path)
    
    # if os.path.exists(output_full_path):
    #     exit("the path exists, please double check")
    
    # TODO (this is for antmaze/pointmaze)
    if args.env=="pointmaze":
        MAX_N_SOL_PER_STL = 4
    elif args.env=="antmaze":
        MAX_N_SOL_PER_STL = 2
    # collect
    uniques = set()
    out_data_list=[]
    output_path = []
    trajs_pool=[]
    us_pool=[]
    all_data_for_stat=[]
    cnt_d={0:0, 1:0, 2:0, 3:0}
    cnt_valids_d={0:0, 1:0, 2:0, 3:0}
    for input_path in input_data_paths:
        if args.env in ["simple", "dubins"]:
            dirs = sorted(os.listdir(ospj(EXP_DIR, input_path)))
        else:
            dirs = [""]
        for sub_dir in dirs:
            np_path = ospj(EXP_DIR, input_path, sub_dir, "data.npz")
            if os.path.exists(np_path):
                in_data = np.load(np_path, allow_pickle=True)['data']
                if len(in_data)>=100:
                    print("IN", input_path, sub_dir, len(in_data), "OUT", len(out_data_list), len(uniques),"ids", cnt_d, cnt_valids_d)
                    for data_i, data_item in enumerate(in_data):
                        if data_i%10000==0 or data_i==len(in_data)-1:
                            print("IN", input_path, sub_dir, len(in_data),data_i)
                        # TODO do sth for ant/pointmaze to reduce data size (maybe limit to 2 sols per stl)
                        if args.env in ["pointmaze", "antmaze"]:
                            # dict_keys(['stl_type_i', 'stl_seed', 'init_cell', 'goal_cell', 'ego', 'goals_indices', 'obstacles_indices', 'trajs', 'obs', 'actions', 'stl', 'objects'])
                            if data_i==0 or in_data[data_i-1]['stl_seed']!=data_item['stl_seed']:
                                # only work on the first on of its seed
                                sub_groups=[in_data[data_i]]
                                for data_j in range(data_i+1, len(in_data)):
                                    # print(in_data[data_j]['stl_seed'], data_item['stl_seed'])
                                    if in_data[data_j]['stl_seed'] != data_item['stl_seed']:
                                        break
                                    sub_groups.append(in_data[data_j])
                                
                                # sub_trajs = torch.from_numpy(
                                #     np.stack([sub_rec['trajs'] for sub_rec in sub_groups], axis=0)
                                # ).float()
                                # stl = data_item['stl']
                                # dict1={}
                                # dict2={}
                                # real_stl = find_ap_in_lines(0, dict1, dict2, stl, numpy=True, real_stl=True, ap_mode="l2", until1=True) 
                                
                                # scores = real_stl(sub_trajs, tau=5000)[:, 0]
                                # valid_indices = torch.where(scores>0)[0]
                                
                                # valid_trajs = sub_trajs[valid_indices, :]
                                # cnt_tmp = 0
                                # if len(valid_indices)==0:
                                #     continue
                                # elif len(valid_indices)==1:
                                #     cnt_tmp = 1
                                #     out_data_list.append(sub_groups[valid_indices[0]])
                                # else:
                                #     valid_trajs_flat = valid_trajs.reshape(valid_trajs.shape[0], -1)
                                #     sim = torch.norm(valid_trajs_flat[:,None]-valid_trajs_flat[None,:], dim=-1)
                                #     nondiag_lower_indices = torch.tril_indices(valid_trajs.shape[0], valid_trajs.shape[0], offset=-1)
                                #     nondiag_values = sim[nondiag_lower_indices[0], nondiag_lower_indices[1]]
                                #     max_at_i = torch.argmax(nondiag_values)
                                #     pairs = nondiag_lower_indices[0, max_at_i], nondiag_lower_indices[1, max_at_i]
                                #     out_data_list.append(sub_groups[valid_indices[pairs[0]]])
                                #     out_data_list.append(sub_groups[valid_indices[pairs[1]]])
                                #     cnt_tmp = 2
                                #     for other_valid_idx in valid_indices:
                                #         if other_valid_idx not in [valid_indices[pairs[0]], valid_indices[pairs[1]]] and cnt_tmp<MAX_N_SOL_PER_STL:
                                #             cnt_tmp+=1
                                #             out_data_list.append(sub_groups[other_valid_idx])
                                for data_j in range(min(MAX_N_SOL_PER_STL, len(sub_groups))):
                                    out_data_list.append(sub_groups[data_j])
                                cnt_d[data_item['stl_type_i']]+=1
                                cnt_valids_d[data_item['stl_type_i']] += min(MAX_N_SOL_PER_STL, len(sub_groups))
                            continue       
                                                
                        out_data_list.append(data_item)                   
                        cnt_d[data_item['stl_type_i']]+=1
                        cnt_valids_d[data_item['stl_type_i']]+=np.sum(data_item['score']>0)
                        
                        if args.env in ["simple", "dubins"]:
                            if 'stl_seed' in data_item and data_item['stl_seed'] not in uniques:
                                uniques.add(data_item['stl_seed'])
                        
                        if args.env in ["panda"]:
                            # dict_keys(['stl_type_i', 'trial_i', 'ego', 'stl', 'objects', 'goals_indices', 'obstacles_indices', 'score', 'us', 'state', 'trajs'])
                            old_args = np.load(ospj(EXP_DIR, input_path, "args.npz"), allow_pickle=True)['args'].item()
                            if hasattr(old_args, "base_seed") and old_args.base_seed is not None:
                                base_seed = old_args.base_seed
                            else:
                                base_seed = old_args.seed * old_args.num_trials * 4
                            
                            if "seed" not in data_item:
                                trial_i = data_item['trial_i']
                                stl_type_i = data_item['stl_type_i']
                                seedseed = base_seed + trial_i * 4 + stl_type_i
                                data_item['seed'] = seedseed
                            
                            # get data stat update
                            acc_indices = np.where(data_item['score']>0)[0]
                            trajs_pool.append(data_item['trajs'][acc_indices])
                            us_pool.append(data_item['us'][acc_indices])
                        
    print(len(out_data_list))
    os.makedirs(output_full_path, exist_ok=True)
    
    if args.env=="dubins":
        ALL_scores = torch.from_numpy(np.stack([data_item["score"] for data_item in out_data_list], axis=0)).float()
        ALL_xs = torch.from_numpy(np.stack([data_item["state"] for data_item in out_data_list], axis=0)).float()
        ALL_us = torch.from_numpy(np.stack([data_item["us"] for data_item in out_data_list], axis=0)).float()
        ALL_us = ALL_us.reshape(ALL_xs.shape[0], ALL_xs.shape[1], ALL_us.shape[-2], ALL_us.shape[-1])
        ALL_trajs = generate_trajectories_dubins(ALL_xs, ALL_us, args.dt, args.v_max)
        
        for data_i,data_item in enumerate(out_data_list):
            data_item["trajs"] = utils.to_np(ALL_trajs[data_i])

        print(ALL_trajs.shape, ALL_scores.shape)
        valid_index1, valid_index2 = torch.where(
            torch.logical_and(
                torch.logical_and(
                    torch.max(torch.abs(ALL_trajs[:, :, :, 0]), dim=2)[0]<=5,
                    torch.max(torch.abs(ALL_trajs[:, :, :, 1]), dim=2)[0]<=5),
                ALL_scores[:,:,0]>0,
                )
            )
        val_trajs_flat = ALL_trajs[valid_index1, valid_index2]
        val_us_flat = ALL_us[valid_index1, valid_index2]
        all_data_for_stat = torch.cat([val_trajs_flat[:,:-1,:], val_us_flat], dim=-1)
        all_data = all_data_for_stat
        print(all_data.shape)
        all_data = all_data.reshape(-1, all_data.shape[-1])
        print(torch.min(all_data, dim=0)[0], torch.max(all_data,dim=0)[0], torch.mean(all_data, dim=0), torch.std(all_data, dim=0))
    
    
    if args.env == "panda":
        trajs_pool = np.concatenate(trajs_pool, axis=0)
        us_pool = np.concatenate(us_pool, axis=0)
        trajs_min, trajs_max, trajs_mean, trajs_std = get_stat(trajs_pool)
        us_min, us_max, us_mean, us_std = get_stat(us_pool)
        trajs_nt_min, trajs_nt_max, trajs_nt_mean, trajs_nt_std = get_stat(get_nt_flatten(trajs_pool))
        us_nt_min, us_nt_max, us_nt_mean, us_nt_std = get_stat(get_nt_flatten(us_pool))
        
        np.savez(os.path.join(output_full_path, "stat.npz"), data={
            "trajs_min": trajs_min, "trajs_max": trajs_max, "trajs_mean": trajs_mean, "trajs_std": trajs_std,
            "us_min": us_min, "us_max": us_max, "us_mean": us_mean, "us_std": us_std,
            "trajs_nt_min": trajs_nt_min, "trajs_nt_max": trajs_nt_max, "trajs_nt_mean": trajs_nt_mean, "trajs_nt_std": trajs_nt_std,
            "us_nt_min": us_nt_min, "us_nt_max": us_nt_max, "us_nt_mean": us_nt_mean, "us_nt_std": us_nt_std,
        })
        with open(os.path.join(output_full_path, "logs.txt"), "w") as f:
            for line in input_data_paths:
                f.write(line+"\n")
        
    
    
    np.savez(ospj(output_full_path, "data.npz"), data=out_data_list)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--env", type=str, choices=['simple', 'dubins', 'pointmaze', 'antmaze', 'panda'], default='simple')
    add("--dt", type=float, default=0.5)
    add("--v_max", type=float, default=2.0)
    t1=time.time()
    # EXP_DIR="../exps_gstl"

    # # #out_data_path = "g0102-075002_BATCH_data_2k"
    # # out_data_path = "g0102-000000_BATCHdata"
    # # input_data_paths=["g0102-075002_BATCH_data4x500x20","g0102-075001_BATCH_data4x500x20"]

    # out_data_path = "g0116-033333_BATCHdubins"
    # input_data_paths=["g0116-033319_BATCH_dubins4x500x10"]

    # out_data_list=[]
    # for input_path in input_data_paths:
    #     dirs = os.listdir(os.path.join(EXP_DIR, input_path))
    #     for d in dirs:
    #         in_data = np.load(os.path.join(EXP_DIR, input_path, d, "data.npz"), allow_pickle=True)['data']
    #         print("IN",input_path,d,len(in_data), "OUT", len(out_data_list))
    #         for data_item in in_data:
    #             out_data_list.append(data_item)

    # print(len(out_data_list))
    # os.makedirs(os.path.join(EXP_DIR, out_data_path), exist_ok=True)
    # np.savez(os.path.join(EXP_DIR, out_data_path, "data.npz"), data=out_data_list)

    args = parser.parse_args()
    main()

    t2 = time.time()
    print("Finished in %.3f seconds"%(t2-t1))