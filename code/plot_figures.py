import os 
from os.path import join as ospj
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import utils

EXP_DIR = utils.get_exp_dir()
exp_paths = {
    'simple': {
        'ours': "g0128-075243_simple_gnn_F",
        'gru': "g0128-075443_simple_gru_F", 
        'trans': "g0128-164526_simple_trans2_F",
        'tree': "g0128-075448_simple_tree_F",
        "goal": "g0128-075450_simple_goal_F",
        "gnn_diffuser": "g0128-194710_simple_gnn_diffusion",
    },
    'dubins': {
        'ours': "g0128-075645_dubins_gnn_F",
        'gru': "g0128-075659_dubins_gru_F",
        'trans': "g0128-225455_dubins_trans2_F",
        'tree': "g0128-080624_dubins_tree_F",
        "goal": "g0128-080648_dubins_goal_F",
        "gnn_diffuser": "g0128-194713_dubins_gnn_diffusion",
    },
    'pointmaze': {
        'ours': "g0128-073754_pointmaze_gnn_F",
        'gru': "g0128-073802_pointmaze_gru_F",
        'trans': "g0128-230116_pointmaze_trans2_F",
        'tree': "g0128-073826_pointmaze_tree_F",
        "goal": "g0128-073910_pointmaze_goal_F",
        "gnn_diffuser": "g0128-194756_pointmaze_gnn_diffusion",
    },
    'antmaze': {
        'ours': "g0128-164357_antmaze_gnn_F",
        'gru': "g0128-164414_antmaze_gru_F",
        'trans': "g0128-230142_antmaze_trans2_F",
        'tree': "g0128-074038_antmaze_tree_F",
        "goal": "g0128-074048_antmaze_goal_F",
        "gnn_diffuser": "g0128-200948_antmaze_gnn_diffusion",
    },
    'panda': {
        'ours': "g0128-084928_panda_gnn_F",
        'gru': "g0128-162749_panda_gru_bfs_F",
        'trans': "g0129-203130_panda_trans2_F",
        'tree': "g0128-084943_panda_tree_F",
        "goal": "g0128-084948_panda_goal_F",
        "gnn_diffuser": "g0128-204325_panda_gnn_diffusion",
    }
}


def try_open_npz(fpath, data_name, key='data'):
    full_path = ospj(EXP_DIR, fpath, data_name)
    if os.path.exists(full_path):
        d = np.load(full_path, allow_pickle=True)[key]
    else:
        d = None
    return d

def main():
    QUIET=True
    for env in exp_paths:
        exp_paths[env]["gnn"] = exp_paths[env]["ours"]
    
    data_d = {}
    for env in ["simple", "dubins", "pointmaze", "antmaze", "panda"]:
        data_d[env] = {}
        for encoder in ["gnn", "ego", "goal", "gru", "trans", "tree", "gnn_diffuser"]:
            # training results
            train_val_acc_file = "stl_acc.npz"
            test_profile_file = "results.npz"
            test_acc_file = "stl_acc.npz"
            data_d[env][encoder] = {}
            if encoder in exp_paths[env]:
                tmp_data = try_open_npz(exp_paths[env][encoder], train_val_acc_file)
                if tmp_data is not None:
                    data_d[env][encoder]["train_val_acc_data"] = tmp_data
                else:
                    if not QUIET:
                        print("%s doesn't have train_val_acc data:%s"%(exp_paths[env][encoder], train_val_acc_file))
            else:
                if not QUIET:
                    print("encoder:%s not in exp_paths[%s], exists are:%s"%(encoder, env, exp_paths[env].keys()))
            
            aug_list = ["results_test_aug%d"%n_augs for n_augs in [1,3,5,8]]
            
            if encoder in exp_paths[env]:
                flow_versions = ["results_test_V%d"%VV for VV in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] + list(range(16,30))]
                if encoder in ["gnn"]:
                    if env in ["simple", "dubins", "panda"]:
                        poss_dirs = ["results_test", "results_test_cem", "results_test_tj"] + ["results_test_tj_%d"%niters for niters in [5, 10, 20, 50, 100, 150, 300, 500]] + flow_versions
                    else:
                        poss_dirs = ["results_test"] + flow_versions
                elif encoder in ["goal", "ego"]:
                    if env in ["simple", "dubins", "panda"]:
                        poss_dirs = ["results_test", "results_test_ctg", "results_test_ltl"]
                    else:
                        poss_dirs = ["results_test", "results_test_ltl"]
                else:
                    poss_dirs = ["results_test"]
                poss_dirs = poss_dirs + aug_list
                for poss_dir in poss_dirs:
                    data_d[env][encoder][poss_dir] = {}
                    tmp_data = try_open_npz(exp_paths[env][encoder]+"/"+poss_dir, test_profile_file)
                    if tmp_data is not None:
                        data_d[env][encoder][poss_dir]["results"] = tmp_data.item()
                    else:
                        if not QUIET:
                            print("%s doesn't have data:%s"%(exp_paths[env][encoder], test_profile_file))
                    
                    tmp_data = try_open_npz(exp_paths[env][encoder]+"/"+poss_dir, test_acc_file)
                    if tmp_data is not None:
                        data_d[env][encoder][poss_dir]["accs"] = tmp_data
                    else:
                        if not QUIET:
                            print("%s doesn't have data:%s"%(exp_paths[env][encoder], test_acc_file))
                    
                    '''
                    >>> acc=np.load("g0128-084928_panda_gnn_F/results_test_V1/stl_acc.npz", allow_pickle=True)['data']
                    >>> acc[0]
                    {'epoch': 0, 'train': np.float64(0.5), 'val': np.float64(0.4296875), 
                    'train_0': np.float64(0.2727272727272727),  'train_1': np.float64(0.4883720930232558), 
                    'train_2': np.float64(0.2692307692307692),  'train_3': np.float64(0.6875), 
                    'val_0': np.float64(0.6666666666666666),  'val_1': np.float64(0.2926829268292683), 
                    'val_2': np.float64(0.4482758620689655),  'val_3': np.float64(0.4897959183673469)}
                    >>> acc[0]['train']
                    np.float64(0.5)
                    >>> d = np.load("g0128-084928_panda_gnn_F/results_test_V1/results.npz", allow_pickle=True)['data']
                    >>> d.item().keys()
                    dict_keys(['train', 'val', 'meta'])
                    >>> d.item()['train'][0].keys()
                    dict_keys(['t', 'scores', 'acc', 'index', 'rec_i', 'stl_i', 'stl_type_i', 'trajs', 'trajs_panda'])
                    '''
                    # compute time, compute acc, for both train and val
                    if "results" in data_d[env][encoder][poss_dir] and "accs" in data_d[env][encoder][poss_dir]:
                        results = data_d[env][encoder][poss_dir]["results"]
                        accs = data_d[env][encoder][poss_dir]["accs"]
                        # discard the first one, because it is normally huge
                        if 'train' in results and len(results['train'])>=1:
                            train_t = np.mean([results['train'][kk]["t"] for kk in range(1, len(results['train']))])
                            train_acc = np.mean([results['train'][kk]["acc"] for kk in range(0, len(results['train']))])
                            train_accs = [accs[0]['train_%d'%type_i] for type_i in range(4)]
                        
                        else:
                            train_t = -1
                            train_acc = -1
                            train_accs = [-1, -1, -1, -1]
                        
                        val_t = np.mean([results['val'][kk]["t"] for kk in range(1, len(results['val']))])
                        val_acc = np.mean([results['val'][kk]["acc"] for kk in range(0, len(results['val']))])
                        val_accs = [accs[0]['val_%d'%type_i] for type_i in range(4)]
                        
                        data_d[env][encoder][poss_dir]['plot'] = {
                            "t":(train_t+val_t)/2, "train_acc": train_acc, "val_acc":val_acc,
                            "train_accs":train_accs, "val_accs": val_accs,
                            }
                        
                        print("Path:%-40s Env:%-9s  encoder:%-15s  dir:%-20s  t_avg:%6.3f  len:%d/%d   train_acc:%.3f (%.3f, %.3f, %.3f, %.3f)   val_acc:%.3f (%.3f, %.3f, %.3f, %.3f)"%(
                            exp_paths[env][encoder], env, encoder, poss_dir, (train_t+val_t)/2, len(results['train']), len(results['val']),
                            train_acc, train_accs[0], train_accs[1], train_accs[2], train_accs[3],
                            val_acc, val_accs[0], val_accs[1], val_accs[2], val_accs[3]
                        ))
    
    env_rename_d = {
        "simple":"Linear",
        "dubins":"Dubins",
        "pointmaze":"PointMaze",
        "antmaze":"AntMaze",
        "panda":"Franka Panda",
    }
    
    # # # # plot main comparison
    # # # # ours, diffusion, LTLDoG, {CTG, cem, gradient}
    plot_main_figure(data_d, env_rename_d)
    
    # # # # plot diff encoder 
    # # # # ours, gru, trans, treelstm, goal
    plot_diff_encoder(data_d, env_rename_d)
    plot_categorized_accs(data_d, env_rename_d)
    
    # # # plot flow pattern
    # # # ours, V1, V2, V3, vs original (maybe also with images?)
    plot_flow_pattern(data_d, env_rename_d)
    plot_ood_test(data_d, env_rename_d)
    np.savez("%s/icml_data_%s.npz"%(EXP_DIR, datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")), data=data_d)
    return

def plot_ood_test(data_d, env_rename_d):
    tmp_dict = {}
    for env in data_d:
        tmp_dict[env] = {}
        for encoder in [ "gru", "trans", "tree", "gnn"]:
            tmp_dict[env][encoder] = []
            for sub_dir in ["results_test", "results_test_aug1", "results_test_aug3", "results_test_aug5", "results_test_aug8"]:
                tmp_dict[env][encoder].append(data_d[env][encoder][sub_dir]['plot']['val_acc'])
    
    colors_d = {
        "goal": "midnightblue", # '#9b59b6',
        "gru": '#3498db', #"salmon",
        "trans": '#2ecc71', #"orange",
        "tree": 'orange',              #"green",
        "gnn": 'crimson',         #'#3498db',   # "blue",
        "ours_v1": 'salmon',         #"#56DEFE", '#5498ff', #"cyan",
    }
    
    name_d = {
        "goal": "Goal-Conditioned",
        "gru": "GRU",
        "trans": "Transformer",
        "tree": "TreeLSTM",
        "gnn": "GNN (Ours)",
        "ours": "GNN (Ours)",
        "ours_v1": "GNN (Fast)",        
    }
    
    x_labels_dict = {
        "simple": [17.3, 24.6, 36.3, 55.2, 104.0],
        "dubins": [17.3, 24.6, 36.3, 55.1, 104.1],
        "pointmaze": [8.881, 11.863, 15.381, 19.453, 26.846],
        "antmaze": [8.779, 11.743, 15.218, 19.493, 27.2],
        "panda": [16.4, 21.817, 28.474, 36.710, 52.914],
    }
    
    x_labels = ["Original", "+1/node", "+3/node", "+5/node", "+8/node"]
    
    MARKERSIZE = 4
    LINE_WIDTH = 2
    # Create figure with (1,5) subplots
    fig, axes = plt.subplots(1, len(tmp_dict), figsize=(15, 1.5))#, sharey=True)

    # Iterate through environments and plot encoders
    ax_i=0
    for ax, (env, encoder_data) in zip(axes, tmp_dict.items()):
        # Add gradually darkening shaded backgrounds
        for i in range(len(x_labels)):
            if i==0:
                print(-0.5, i+0.5)
                ax.axvspan(i-0.5, i + 0.5, color=(0.5, 0.9 - i * 0.1, 0.7), alpha=0.3, label="Original STL" if ax_i==0 else None)
            elif i==4:
                print(i-0.5, 4)
                ax.axvspan(i-0.5, i+0.5, color=(0.9 - i * 0.05, 0.9 - i * 0.15, 0.9 - i * 0.15), alpha=0.3, label="OOD STL" if ax_i==0 else None)
            else:
                print(i - 0.5, i + 0.5)
                ax.axvspan(i - 0.5, i + 0.5, color=(0.9 - i * 0.05, 0.9 - i * 0.15, 0.9 - i * 0.15), alpha=0.3)

        for encoder, perf_values in encoder_data.items():
            ax.plot(x_labels, perf_values, color=colors_d[encoder], label=name_d[encoder], marker="s", linestyle="--", markersize=MARKERSIZE, linewidth=LINE_WIDTH)
        # Formatting
        ax.set_title(env_rename_d[env].capitalize(), fontsize=10)
        ax.set_xticklabels(["%d/%d"%(x_labels_dict[env][0], xxx-x_labels_dict[env][0]) for xxx in x_labels_dict[env]], fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(-0.5, 4.5)
        ax_i+=1

    # Set common y-axis label
    axes[0].set_ylabel("STL satisfaction", fontsize=10)
    
    axes[2].set_xlabel("Average number of nodes per STL (original/augmented)", fontsize=10)

    # Add legend outside the plot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=6, fontsize=10)#, title="Encoders")
    plt.savefig('%s/%s'%(EXP_DIR, "z_icml_fig5_ood.pdf"), bbox_inches='tight', pad_inches=0.01, metadata={})
    plt.close()
    return


def plot_main_figure(data_d, env_rename_d):
    # env_list=["simple", ]
    PLOT_TRAIN = False
    BAR_WIDTH = 0.75
    OPACITY = 1
    colors = {
        "trajopt_5": "lightgrey",
        "trajopt_10": "lightgrey",
        "trajopt_20": "grey",
        "trajopt_50": "grey",
        "trajopt_100": "slategray",
        "trajopt_150": "slategray",
        
        "trajopt_light": "lightgray",
        "trajopt_heavy": "slategray",
        
        "cem": "midnightblue",        
        
        "ctg": '#3498db', 
        "ltl": '#2ecc71',
        
        "diffuse": 'orange',              
        "ours": 'crimson', 
        "ours_v1": 'salmon',         
        "ours_v2": "royalblue",
        "ours_v3": "darkblue",
    }
    
    name_d = {
        "trajopt_5": "Trajopt(5)",
        "trajopt_10": "Trajopt(10)",
        "trajopt_20": "Trajopt(20)",
        "trajopt_50": "Trajopt(50)",
        "trajopt_100": "Trajopt(100)",
        "trajopt_150": "Trajopt(150)",
        
        "trajopt_light": "Grad-lite",
        "trajopt_heavy": "Grad",
        
        "cem": "CEM",
        "ctg": "CTG",
        "ltl": "LTLDoG",
        "diffuse": "TeLoGraD",
        
        "ours": "TeLoGraF",
        "ours_v1": "TeLoGraF (Fast)",
        "ours_v2": "Ours_v2",
        "ours_v3": "Ours_v3",
        
    }
    env_d = env_rename_d    
    # ours, diffusion, LTLDoG, {CTG, cem, gradient}
    trajopts_list = ["results_test_tj_%d"%niters for niters in [5, 10, 20, 50, 100, 150, 300, 500]]
    poss_dirs = trajopts_list + ["results_test_cem", "diffusion", "ours"]
    
    if PLOT_TRAIN:
        fig = plt.figure(figsize=(10, 4.5))
        gs = fig.add_gridspec(3, 5)
    else:
        fig = plt.figure(figsize=(10, 3.5))
        gs = fig.add_gridspec(2, 5)
    showed_set = set()
    
    for env_idx, env in enumerate(data_d.keys()):            
        # encoder, subdir, rename        
        query_list=[
            # tj
            ("gnn", "results_test_tj_50", "trajopt_50", 0) if env=="panda" else ("gnn", "results_test_tj_10", "trajopt_10", 0),
            ("gnn", "results_test_tj_150", "trajopt_150", 1) if env=="panda" else ("gnn", "results_test_tj_50", "trajopt_50", 1),
           
            # cem
            ("gnn", "results_test_cem", "cem", 2),
            
            # ctg
            ("goal", "results_test_ctg", "ctg", 3),
            
            # ltldog
            ("goal", "results_test_ltl", "ltl", 4),
            
            # diffuse
            ("gnn_diffuser", "results_test", "diffuse", 5),
            
            # ours
            ("gnn", "results_test", "ours", 6),
            ("gnn", "results_test_V1", "ours_v1", 7),
        ]
        methods=[]
        t_list=[]
        train_acc_list=[]
        val_acc_list=[]
        pos_list=[]
        
        for enc, sub, rename, label_pos_i in query_list:
            if enc in data_d[env] and sub in data_d[env][enc] and "plot" in data_d[env][enc][sub]:
                if "trajopt" in rename:
                    if env=="panda":
                        mapping = {"trajopt_50": "trajopt_light", "trajopt_150": "trajopt_heavy"}
                    else:
                        mapping = {"trajopt_10": "trajopt_light", "trajopt_50": "trajopt_heavy"}
                    methods.append(mapping[rename])
                else:
                    methods.append(rename)
                
                plot_data=data_d[env][enc][sub]["plot"]
                t_list.append(plot_data['t'])
                train_acc_list.append(plot_data['train_acc'])
                val_acc_list.append(plot_data['val_acc'])
                pos_list.append(label_pos_i)
        
        if PLOT_TRAIN:
            num_rows = 3
        else:
            num_rows = 2
        for row_i in range(num_rows):        
            ax = fig.add_subplot(gs[row_i, env_idx])
            x = np.arange(len(methods))
            for method_idx, method in enumerate(methods):
                method_name = name_d[method]
                if PLOT_TRAIN:
                    if row_i==0:
                        value_to_put = train_acc_list[method_idx]
                    elif row_i==1:
                        value_to_put = val_acc_list[method_idx]
                    else:
                        value_to_put = t_list[method_idx]
                else:
                    if row_i==0:
                        value_to_put = val_acc_list[method_idx]
                    else:
                        value_to_put = t_list[method_idx]
                pos_i = x[method_idx]
                ax.bar(pos_i, value_to_put, BAR_WIDTH, alpha=OPACITY, color=colors[method], label=f'{method_name}' if method_name not in showed_set else None)
                showed_set.add(method_name)

            if row_i == 0:
                if env_idx==0:
                    ax.set_ylabel('STL satisfaction')
            else:
                if env_idx==0:
                    if PLOT_TRAIN:
                        if row_i==1:
                            ax.set_ylabel('STL satisfaction')
                        else:
                            ax.set_ylabel('Runtime (s)')
                    else:
                        ax.set_ylabel('Runtime (s)')
                if row_i == num_rows-1:
                    ax.set_xlabel(env_d[env].capitalize())
            ax.set_xticks([])
            # Only show legend for the last subplot
            if env_idx == 2 and row_i==0:
                fig.legend(bbox_to_anchor=(0.5, 1.065), loc='upper center', ncol=8)

    plt.tight_layout()
    plt.savefig('%s/%s'%(EXP_DIR, "z_icml_fig1_main.pdf"), bbox_inches='tight', pad_inches=0.05, metadata={})
    plt.close()


def plot_diff_encoder(data_d, env_rename_d):
    BAR_WIDTH = 0.4
    OPACITY = 1
    colors = {
        "goal": "midnightblue", 
        "gru": '#3498db', 
        "trans": '#2ecc71',
        "tree": 'orange', 
        "ours": 'crimson', 
        "ours_v1": 'salmon',
    }
    
    name_d = {
        "goal": "Goal-Conditioned",
        "gru": "GRU",
        "trans": "Transformer",
        "tree": "TreeLSTM",
        
        "ours": "GNN (Ours)",
        "ours_v1": "GNN (Fast)",        
    }
    
    env_d = env_rename_d
    
    # ours, diffusion, LTLDoG, {CTG, cem, gradient}    
    showed_set = set()
    
    fig = plt.figure(figsize=(10, 2.0))
    gs = fig.add_gridspec(1, 5)
    
    showed_set = set()
    
    for env_idx, env in enumerate(data_d.keys()):            
        # encoder, subdir, rename        
        query_list=[
            # tj       
            ("gru", "results_test", "gru", 0),
            ("trans", "results_test", "trans", 1),
            ("tree", "results_test", "tree", 2),
            ("gnn", "results_test", "ours", 3),
        ]
        methods=[]
        t_list=[]
        train_acc_list=[]
        val_acc_list=[]
        pos_list=[]
    
        for enc, sub, rename, label_pos_i in query_list:
            if enc in data_d[env] and sub in data_d[env][enc] and "plot" in data_d[env][enc][sub]:
                methods.append(rename)
                plot_data=data_d[env][enc][sub]["plot"]
                t_list.append(plot_data['t'])
                train_acc_list.append(plot_data['train_acc'])
                val_acc_list.append(plot_data['val_acc'])
                pos_list.append(label_pos_i)
        
        
        for row_i in range(1):        
            ax = fig.add_subplot(gs[row_i, env_idx])
            x = np.arange(len(methods))
            for method_idx, method in enumerate(methods):
                method_name = name_d[method]
                print(env_idx, row_i)
                if row_i==0:
                    ax.bar(x[method_idx] - BAR_WIDTH/2, train_acc_list[method_idx], BAR_WIDTH,
                        alpha=OPACITY, color=colors[method], label=f'{method_name}_train' if env_idx==0 else None)
                    ax.bar(x[method_idx] + BAR_WIDTH/2, val_acc_list[method_idx], BAR_WIDTH,
                            alpha=OPACITY, color=colors[method], hatch='//', label=f'{method_name}_val' if env_idx==0 else None)
                else:
                    value_to_put = t_list[method_idx]
                    pos_i = x[method_idx]
                    ax.bar(pos_i, value_to_put, BAR_WIDTH, alpha=OPACITY, color=colors[method])
                showed_set.add(method_name)
            
            # Customize subplot
            if row_i == 0:
                if env_idx==0:
                    ax.set_ylabel('STL satisfaction')
                ax.set_xlabel(env_d[env].capitalize())
            else:
                if env_idx==0:
                    ax.set_ylabel('Runtime (s)')
                ax.set_xlabel(env_d[env].capitalize())
            
            ax.set_xticks([])

            # Only show legend for the last subplot
            if env_idx == 2 and row_i==0:
                fig.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=4)

    plt.tight_layout()
    plt.savefig('%s/%s'%(EXP_DIR, "z_icml_fig2_encoder.pdf"), bbox_inches='tight', pad_inches=0.05, metadata={})
    plt.close()

def plot_categorized_accs(data_d, env_rename_d):
    colors_d = {
        "goal": "midnightblue", 
        "gru": '#3498db',
        "trans": '#2ecc71', 
        "tree": 'orange',
        "gnn": 'crimson',
        "ours_v1": 'salmon',
    }
    rename_d={"gru":"GRU", "trans":"Transformer", "tree":"TreeLSTM", "gnn":"GNN"}
    encoders = ["gru", "trans", "tree", "gnn"]
    subdir = "results_test"
    for env in data_d:
        for encoder in encoders:
            print("%-10s %-12s %.4f (%.3f,%.3f,%.3f,%.3f) %.4f (%.3f,%.3f,%.3f,%.3f) "%(
                env, encoder, 
                data_d[env][encoder][subdir]['plot']['train_acc'],
                data_d[env][encoder][subdir]['plot']['train_accs'][0],
                data_d[env][encoder][subdir]['plot']['train_accs'][1],
                data_d[env][encoder][subdir]['plot']['train_accs'][2],
                data_d[env][encoder][subdir]['plot']['train_accs'][3],
                
                data_d[env][encoder][subdir]['plot']['val_acc'],
                data_d[env][encoder][subdir]['plot']['val_accs'][0],
                data_d[env][encoder][subdir]['plot']['val_accs'][1],
                data_d[env][encoder][subdir]['plot']['val_accs'][2],
                data_d[env][encoder][subdir]['plot']['val_accs'][3],
                
            ))

    env_d = env_rename_d
    data = {
            "gru":data_d[env]["gru"][subdir]['plot']['val_accs'],
            "trans":data_d[env]["trans"][subdir]['plot']['val_accs'],
            "tree":data_d[env]["tree"][subdir]['plot']['val_accs'],
            "gnn":data_d[env]["gnn"][subdir]['plot']['val_accs'],
                }
    
    # Create figure
    fig, axes = plt.subplots(1, 5, figsize=(16, 3), subplot_kw=dict(polar=True))
    for env_i, (env, ax) in enumerate(zip(data_d,axes)):
        data = {
            "gru":data_d[env]["gru"][subdir]['plot']['val_accs'],
            "trans":data_d[env]["trans"][subdir]['plot']['val_accs'],
            "tree":data_d[env]["tree"][subdir]['plot']['val_accs'],
            "gnn":data_d[env]["gnn"][subdir]['plot']['val_accs'],
                }
        attribute_labels = ["I", "II", "III", "IV"]  # Categories (axes)
        num_vars = len(attribute_labels)  # Number of attributes (axes)
        # Compute angle for each axis
        angles = (np.linspace(0, 2 * np.pi, num_vars, endpoint=False)).tolist()
        angles += angles[:1]  # Close the radar chart shape
        # Define colors
        colors = [colors_d[key] for key in encoders]
        # Plot each method
        for (method, values), color in zip(data.items(), colors):
            values += values[:1]  # Close shape
            ax.plot(angles, values, color=color, linewidth=2, linestyle="-", label=rename_d[method])
            ax.fill(angles, values, color=color, alpha=0.2)  # Fill the shape
        # Set attribute labels    
        ax.set_xticks(angles[:-1])
        if env_i==0:
            xticks = ax.get_xticks()  # Get angles of labels (radians)
            xticklabels = ax.set_xticklabels(attribute_labels, fontsize=12)
        else:
            ax.set_xticklabels([])

        if env_i==2:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), fontsize=12, ncol=4, frameon=True)
        ax.set_title(env_d[env].capitalize(), pad=30)
        ax.set_facecolor("#f5f5f5")  # Light gray background    
    plt.savefig('%s/%s' % (EXP_DIR, "z_icml_fig3_radar.pdf"), bbox_inches='tight', dpi=300, pad_inches=0.01, metadata={})
    plt.close()

def plot_flow_pattern(data_d, env_rename_d):
    env = "dubins"
    encoder = "gnn"
    gaps=[100, 50, 33, 25, 20, 10, 5, 3, 2, 1]
    sub_dir_list = ["results_test_V%d"%(dd) for dd in range(14, 24)]
    t_list=[]
    train_acc_list=[]
    val_acc_list=[]
    for sub_dir in sub_dir_list:
        tmp_data = data_d[env][encoder][sub_dir]
        t_list.append(tmp_data['plot']['t'])
        train_acc_list.append(tmp_data['plot']['train_acc'])
        val_acc_list.append(tmp_data['plot']['val_acc'])
        print(sub_dir, t_list[-1], train_acc_list[-1], val_acc_list[-1])
    
    xs = gaps
    
    # Create figure
    FONTSIZE = 14
    LINE_WIDTH = 3
    MARKERSIZE = 6
    fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)  # Share x-axis

    # === Subplot 1: STL Satisfaction ===
    axes[0].plot(xs, train_acc_list, color="red", label="Train", marker="o", linestyle="-", markersize=MARKERSIZE, linewidth=LINE_WIDTH)
    axes[0].plot(xs, val_acc_list, color="green", label="Val", marker="s", linestyle="--", markersize=MARKERSIZE, linewidth=LINE_WIDTH)
    axes[0].set_ylabel("STL satisfaction", fontsize=FONTSIZE)
    axes[0].grid(True, linestyle="--", alpha=0.5)  # Add light grid
    if env=="dubins":
        axes[0].legend(fontsize=FONTSIZE,bbox_to_anchor=(0.9, 0.4))  # Move legend to bottom-right
    else:
        axes[0].legend(fontsize=FONTSIZE, loc="best", bbox_to_anchor=(0.9, 0.6))  # Move legend to bottom-right
    axes[0].set_facecolor("#f5f5f5")  # Light gray background

    # Vertical Lines & Labels
    axes[0].axvline(x=10, color="black", linestyle="--", linewidth=1.5)  # Ours
    axes[0].axvline(x=100, color="black", linestyle="--", linewidth=1.5)  # Ours-full
    if env=="dubins":
        axes[0].text(10+1, 0.16, "TeLoGraF (Fast)", fontsize=FONTSIZE-2, verticalalignment="bottom", horizontalalignment="left") #, rotation=90)
        axes[0].text(100-17, 0.16, "TeLoGraF", fontsize=FONTSIZE-2, verticalalignment="bottom", horizontalalignment="left")#, rotation=90)
    else:
        axes[0].text(10+1, 0.24, "TeLoGraF (Fast)", fontsize=FONTSIZE-2, verticalalignment="bottom", horizontalalignment="left") #, rotation=90)
        axes[0].text(100-17, 0.24, "TeLoGraF", fontsize=FONTSIZE-2, verticalalignment="bottom", horizontalalignment="left")#, rotation=90)

    # === Subplot 2: Runtime ===
    axes[1].plot(xs, t_list, color="blue", marker="d", linestyle="-.", markersize=MARKERSIZE, linewidth=LINE_WIDTH)
    axes[1].set_ylabel("Runtime (s)", fontsize=FONTSIZE)
    axes[1].set_xlabel("ODE Steps", fontsize=FONTSIZE)
    axes[1].grid(True, linestyle="--", alpha=0.5)  # Add light grid
    axes[1].set_facecolor("#f5f5f5")  # Light gray background

    # === Formatting ===
    plt.tight_layout()
    plt.savefig('%s/%s' % (EXP_DIR, "z_icml_fig4_flow.pdf"), bbox_inches='tight', dpi=300, pad_inches=0.05, metadata={})
    plt.close()
    return


if __name__ == "__main__":
    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.3f seconds"%(t2-t1))