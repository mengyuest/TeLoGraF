#### graph-perturbation test (for each env/arch, we test for --max_augs=1/3/5/8)
#######################
# env-0 simple
# Ours
python train_gstl_v1.py --env simple --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075243_simple_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env simple --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075243_simple_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env simple --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075243_simple_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env simple --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075243_simple_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only

# GRU
python train_gstl_v1.py --env simple --encoder gru --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-075443_simple_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env simple --encoder gru --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-075443_simple_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env simple --encoder gru --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-075443_simple_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env simple --encoder gru --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-075443_simple_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only

# Transformer
python train_gstl_v1.py --env simple --encoder trans --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-164526_simple_trans2_F --nlayers 2 --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env simple --encoder trans --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-164526_simple_trans2_F --nlayers 2 --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env simple --encoder trans --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-164526_simple_trans2_F --nlayers 2 --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env simple --encoder trans --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-164526_simple_trans2_F --nlayers 2 --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only

# TreeLSTM
python train_gstl_v1.py --env simple --encoder tree_lstm --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-075448_simple_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env simple --encoder tree_lstm --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-075448_simple_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env simple --encoder tree_lstm --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-075448_simple_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env simple --encoder tree_lstm --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-075448_simple_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only


#######################
# env-1 dubins
# Ours
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only

# GRU
python train_gstl_v1.py --env dubins --encoder gru --flow --first_sat_init --skip_first_eval  --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-075659_dubins_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env dubins --encoder gru --flow --first_sat_init --skip_first_eval  --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-075659_dubins_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env dubins --encoder gru --flow --first_sat_init --skip_first_eval  --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-075659_dubins_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env dubins --encoder gru --flow --first_sat_init --skip_first_eval  --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-075659_dubins_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only

# Transformer
python train_gstl_v1.py -e dubins_trans2_F --env dubins --encoder trans --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2 -T g0128-225455_dubins_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py -e dubins_trans2_F --env dubins --encoder trans --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2 -T g0128-225455_dubins_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py -e dubins_trans2_F --env dubins --encoder trans --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2 -T g0128-225455_dubins_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py -e dubins_trans2_F --env dubins --encoder trans --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2 -T g0128-225455_dubins_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only

# TreeLSTM
python train_gstl_v1.py --env dubins --encoder tree_lstm --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn  -T g0128-080624_dubins_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env dubins --encoder tree_lstm --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn  -T g0128-080624_dubins_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env dubins --encoder tree_lstm --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn  -T g0128-080624_dubins_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env dubins --encoder tree_lstm --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn  -T g0128-080624_dubins_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only


#######################
# env-2 pointmaze
# Ours
python train_gstl_v1.py --env pointmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-073754_pointmaze_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env pointmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-073754_pointmaze_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env pointmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-073754_pointmaze_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env pointmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-073754_pointmaze_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only

# GRU
python train_gstl_v1.py --env pointmaze --encoder gru --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-073802_pointmaze_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env pointmaze --encoder gru --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-073802_pointmaze_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env pointmaze --encoder gru --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-073802_pointmaze_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env pointmaze --encoder gru --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-073802_pointmaze_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only

# Transformer
python train_gstl_v1.py -e pointmaze_trans2_F --env pointmaze --encoder trans --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2 -T g0128-230116_pointmaze_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py -e pointmaze_trans2_F --env pointmaze --encoder trans --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2 -T g0128-230116_pointmaze_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py -e pointmaze_trans2_F --env pointmaze --encoder trans --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2 -T g0128-230116_pointmaze_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py -e pointmaze_trans2_F --env pointmaze --encoder trans --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2 -T g0128-230116_pointmaze_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only

# TreeLSTM
python train_gstl_v1.py --env pointmaze --encoder tree_lstm --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-073826_pointmaze_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env pointmaze --encoder tree_lstm --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-073826_pointmaze_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env pointmaze --encoder tree_lstm --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-073826_pointmaze_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env pointmaze --encoder tree_lstm --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-073826_pointmaze_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only


#######################
# env-3 antmaze
# Ours
python train_gstl_v1.py --env antmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-164357_antmaze_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env antmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-164357_antmaze_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env antmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-164357_antmaze_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env antmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-164357_antmaze_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only

# GRU
python train_gstl_v1.py --env antmaze --encoder gru --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-164414_antmaze_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env antmaze --encoder gru --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-164414_antmaze_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env antmaze --encoder gru --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-164414_antmaze_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env antmaze --encoder gru --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-164414_antmaze_gru_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only

# Transformer
python train_gstl_v1.py -e antmaze_trans2_F --env antmaze --encoder trans --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2 -T g0128-230142_antmaze_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py -e antmaze_trans2_F --env antmaze --encoder trans --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2 -T g0128-230142_antmaze_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py -e antmaze_trans2_F --env antmaze --encoder trans --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2 -T g0128-230142_antmaze_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py -e antmaze_trans2_F --env antmaze --encoder trans --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2 -T g0128-230142_antmaze_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only

# TreeLSTM
python train_gstl_v1.py --env antmaze --encoder tree_lstm --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-074038_antmaze_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env antmaze --encoder tree_lstm --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-074038_antmaze_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env antmaze --encoder tree_lstm --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-074038_antmaze_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env antmaze --encoder tree_lstm --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-074038_antmaze_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only


#######################
# env-4 panda
# Ours
python train_gstl_v1.py --env panda --encoder gnn --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-084928_panda_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env panda --encoder gnn --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-084928_panda_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env panda --encoder gnn --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-084928_panda_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env panda --encoder gnn --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-084928_panda_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only

# GRU
python train_gstl_v1.py --env panda --encoder tree_lstm --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-084943_panda_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env panda --encoder tree_lstm --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-084943_panda_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env panda --encoder tree_lstm --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-084943_panda_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env panda --encoder tree_lstm --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-084943_panda_tree_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only

# Transformer
python train_gstl_v1.py --env panda --encoder gru --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --bfs_encoding -T g0128-162749_panda_gru_bfs_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env panda --encoder gru --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --bfs_encoding -T g0128-162749_panda_gru_bfs_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env panda --encoder gru --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --bfs_encoding -T g0128-162749_panda_gru_bfs_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env panda --encoder gru --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --bfs_encoding -T g0128-162749_panda_gru_bfs_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only

# TreeLSTM
python train_gstl_v1.py --env panda --encoder trans --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --nlayers 2 --hashimoto -T g0129-203130_panda_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug1 --aug_graph --max_aug 1 --val_eval_only
python train_gstl_v1.py --env panda --encoder trans --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --nlayers 2 --hashimoto -T g0129-203130_panda_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug3 --aug_graph --max_aug 3 --val_eval_only
python train_gstl_v1.py --env panda --encoder trans --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --nlayers 2 --hashimoto -T g0129-203130_panda_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug5 --aug_graph --max_aug 5 --val_eval_only
python train_gstl_v1.py --env panda --encoder trans --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --nlayers 2 --hashimoto -T g0129-203130_panda_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _aug8 --aug_graph --max_aug 8 --val_eval_only
