# Training scripts

#######################
# env-0 simple
# Ours/GRU/Transformer/TreeLSTM/Goal-conditioned
python train_gstl_v1.py --env simple --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4
python train_gstl_v1.py --env simple --encoder gru --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto
python train_gstl_v1.py --env simple --encoder trans --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto 
python train_gstl_v1.py --env simple --encoder tree_lstm --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn 
python train_gstl_v1.py --env simple --encoder goal --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4
# train diffusion
python train_gstl_v1.py -e simple_gnn_diffusion --env simple --encoder gnn --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4


#######################
# env-1 dubins
# Ours/GRU/Transformer/TreeLSTM/Goal-conditioned
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4
python train_gstl_v1.py --env dubins --encoder gru --flow --first_sat_init --skip_first_eval  --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto
python train_gstl_v1.py -e dubins_trans2_F --env dubins --encoder trans --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2
python train_gstl_v1.py --env dubins --encoder tree_lstm --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn 
python train_gstl_v1.py --env dubins --encoder goal --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4
# train diffusion
python train_gstl_v1.py -e dubins_gnn_diffusion --env dubins --encoder gnn --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4


#######################
# env-2 pointmaze
# Ours/GRU/Transformer/TreeLSTM/Goal-conditioned
python train_gstl_v1.py --env pointmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4
python train_gstl_v1.py --env pointmaze --encoder gru --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto
python train_gstl_v1.py -e pointmaze_trans2_F --env pointmaze --encoder trans --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2
python train_gstl_v1.py --env pointmaze --encoder tree_lstm --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn
python train_gstl_v1.py --env pointmaze --encoder goal --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4
# train diffusion
python train_gstl_v1.py -e pointmaze_gnn_diffusion --env pointmaze --encoder gnn --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4


#######################
# env-3 antmaze
# Ours/GRU/Transformer/TreeLSTM/Goal-conditioned
python train_gstl_v1.py --env antmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4
python train_gstl_v1.py --env antmaze --encoder gru --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto
python train_gstl_v1.py -e antmaze_trans2_F --env antmaze --encoder trans --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2
python train_gstl_v1.py --env antmaze --encoder tree_lstm --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn
python train_gstl_v1.py --env antmaze --encoder goal --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4
# train diffusion
python train_gstl_v1.py -e antmaze_gnn_diffusion --env antmaze --encoder gnn --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4


#######################
# env-4 panda
# Ours/GRU/Transformer/TreeLSTM/Goal-conditioned
python train_gstl_v1.py --env panda --encoder gnn --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4
python train_gstl_v1.py --env panda --encoder gru --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto
python train_gstl_v1.py --env panda --encoder trans --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto
python train_gstl_v1.py --env panda --encoder tree_lstm --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn
python train_gstl_v1.py --env panda --encoder goal --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4
# --nlayers 2
python train_gstl_v1.py -e panda_trans2_F --env panda --encoder trans --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --nlayers 2 --hashimoto
python train_gstl_v1.py -e panda_gru_bfs_F --env panda --encoder gru --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --bfs_encoding
# train diffusion
python train_gstl_v1.py -e panda_gnn_diffusion --env panda --encoder gnn --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4



####################################################################################
####################################################################################
# Train score predictors (for LTLDoG); for each env, train varied archs  
# env-0 simple
# Ours/GRU/Transformer/TreeLSTM/Goal-conditioned
python train_gstl_v1.py -e score_simple_gnn --epochs 50 --env simple --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier
python train_gstl_v1.py -e score_simple_gru --epochs 50 --env simple --encoder gru --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier --hashimoto
python train_gstl_v1.py -e score_simple_trans --epochs 50 --env simple --encoder trans --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier --hashimoto
python train_gstl_v1.py -e score_simple_tree --epochs 50 --env simple --encoder tree_lstm --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier --hashi_gnn
python train_gstl_v1.py -e score_simple_goal --epochs 50 --env simple --encoder goal --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier

# env-1 dubins
# Ours/GRU/Transformer/TreeLSTM/Goal-conditioned
python train_gstl_v1.py -e score_dubins_gnn --epochs 50 --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier
python train_gstl_v1.py -e score_dubins_gru --epochs 50 --env dubins --encoder gru --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier --hashimoto
python train_gstl_v1.py -e score_dubins_trans --epochs 50 --env dubins --encoder trans --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier --hashimoto
python train_gstl_v1.py -e score_dubins_tree --epochs 50 --env dubins --encoder tree_lstm --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier --hashi_gnn
python train_gstl_v1.py -e score_dubins_goal --epochs 50 --env dubins --encoder goal --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier

# env-2 pointmaze
# Ours/GRU/Transformer/TreeLSTM/Goal-conditioned
python train_gstl_v1.py -e score_pointmaze_gnn --epochs 50 --env pointmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier
python train_gstl_v1.py -e score_pointmaze_gru --epochs 50 --env pointmaze --encoder gru --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier --hashimoto
python train_gstl_v1.py -e score_pointmaze_trans --epochs 50 --env pointmaze --encoder trans --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier --hashimoto
python train_gstl_v1.py -e score_pointmaze_tree --epochs 50 --env pointmaze --encoder tree_lstm --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier --hashi_gnn
python train_gstl_v1.py -e score_pointmaze_goal --epochs 50 --env pointmaze --encoder goal --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier

# env-3 antmaze
# Ours/GRU/Transformer/TreeLSTM/Goal-conditioned
python train_gstl_v1.py -e score_antmaze_gnn --epochs 50 --env antmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier
python train_gstl_v1.py -e score_antmaze_gru --epochs 50 --env antmaze --encoder gru --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier --hashimoto
python train_gstl_v1.py -e score_antmaze_trans --epochs 50 --env antmaze --encoder trans --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier --hashimoto
python train_gstl_v1.py -e score_antmaze_tree --epochs 50 --env antmaze --encoder tree_lstm --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier --hashi_gnn
python train_gstl_v1.py -e score_antmaze_goal --epochs 50 --env antmaze --encoder goal --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier

# env-4 panda
# Ours/GRU/Transformer/TreeLSTM/Goal-conditioned
python train_gstl_v1.py -e score_panda_gnn --epochs 50 --env panda --encoder gnn --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier
python train_gstl_v1.py -e score_panda_gru --epochs 50 --env panda --encoder gru --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier --hashimoto
python train_gstl_v1.py -e score_panda_trans --epochs 50 --env panda --encoder trans --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier --hashimoto
python train_gstl_v1.py -e score_panda_tree --epochs 50 --env panda --encoder tree_lstm --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier --hashi_gnn
python train_gstl_v1.py -e score_panda_goal --epochs 50 --env panda --encoder goal --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --train_classifier