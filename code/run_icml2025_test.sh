#######################
# env-0 simple
# Different archs: Ours/GRU/Transformer/TreeLSTM/Goal-conditioned
python train_gstl_v1.py --env simple --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075243_simple_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env simple --encoder gru --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-075443_simple_gru_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env simple --encoder trans --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-164526_simple_trans2_F --nlayers 2 --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env simple --encoder tree_lstm --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-075448_simple_tree_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env simple --encoder goal --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075450_simple_goal_F --fix --num_evals 128 -b 1 --test_muls 1024

# fast-flow
python train_gstl_v1.py --env simple --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075243_simple_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _V1 -V 1

# CTG
python train_gstl_v1.py --env simple --encoder goal --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075450_simple_goal_F --guidance --guidance_before 10 --guidance_lr 0.2 --guidance_steps 3 --smoothing_factor 500 --guidance_scale 1.0 --fix --suffix ctg --num_evals 128 -b 1 --test_muls 1024 

# LTL
python train_gstl_v1.py --env simple --encoder goal --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075450_simple_goal_F --cls_guidance -C g0129-094245_score_simple_gnn --guidance_before 10 --guidance_lr 0.2 --guidance_steps 3 --smoothing_factor 500 --guidance_scale 1.0 --fix --suffix ltl --num_evals 128 -b 1 --test_muls 1024 

# CEM
python train_gstl_v1.py --env simple --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075243_simple_gnn_F --cem --suffix cem --fix --num_evals 128 -b 1 

# Grad (iters=10/50)
python train_gstl_v1.py --env simple --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075243_simple_gnn_F --trajopt --suffix tj_10 --fix --num_evals 128 -b 1 --trajopt_niters 10
python train_gstl_v1.py --env simple --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075243_simple_gnn_F --trajopt --suffix tj_50 --fix --num_evals 128 -b 1 --trajopt_niters 50

# diffusion (g0128-194710_simple_gnn_diffusion)
python train_gstl_v1.py -e simple_gnn_diffusion --env simple --encoder gnn --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-194710_simple_gnn_diffusion --fix --num_evals 128 -b 1 --test_muls 1024


#######################
# env-1 dubins
# Different archs: Ours/GRU/Transformer/TreeLSTM/Goal-conditioned
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env dubins --encoder gru --flow --first_sat_init --skip_first_eval  --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-075659_dubins_gru_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py -e dubins_trans2_F --env dubins --encoder trans --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2 -T g0128-225455_dubins_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env dubins --encoder tree_lstm --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn  -T g0128-080624_dubins_tree_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env dubins --encoder goal --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-080648_dubins_goal_F --fix --num_evals 128 -b 1 --test_muls 1024

# fast-flow
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _V1 -V 1

# CTG
python train_gstl_v1.py --env dubins --encoder goal --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-080648_dubins_goal_F --guidance --guidance_before 10 --guidance_lr 0.1 --guidance_steps 3 --smoothing_factor 500 --guidance_scale 1.0 --fix --suffix ctg --num_evals 128 -b 1 --test_muls 1024

# LTLDoG
python train_gstl_v1.py --env dubins --encoder goal --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-080648_dubins_goal_F --cls_guidance  -C g0129-104827_score_dubins_gnn --guidance_before 10 --guidance_lr 0.1 --guidance_steps 3 --smoothing_factor 500 --guidance_scale 1.0 --fix --suffix ltl --num_evals 128 -b 1 --test_muls 1024

# CEM
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --cem --suffix cem --fix --num_evals 128 -b 1

# Grad (iters=10/50)
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --trajopt --suffix tj_10 --fix --num_evals 128 -b 1 --trajopt_lr 3e-2 --trajopt_niters 10
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --trajopt --suffix tj_50 --fix --num_evals 128 -b 1 --trajopt_lr 3e-2 --trajopt_niters 50

# diffusion (g0128-194713_dubins_gnn_diffusion)
python train_gstl_v1.py -e dubins_gnn_diffusion --env dubins --encoder gnn --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-194713_dubins_gnn_diffusion --fix --num_evals 128 -b 1 --test_muls 1024


# dubins different flow-test
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _V14 -V 14
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _V15 -V 15
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _V16 -V 16
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _V17 -V 17
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _V18 -V 18
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _V19 -V 19
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _V20 -V 20
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _V21 -V 21
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _V22 -V 22
python train_gstl_v1.py --env dubins --encoder gnn --flow --first_sat_init --skip_first_eval --clip_max 50000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-075645_dubins_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _V23 -V 23


#######################
# env-2 pointmaze
# Different archs: Ours/GRU/Transformer/TreeLSTM/Goal-conditioned
python train_gstl_v1.py --env pointmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-073754_pointmaze_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env pointmaze --encoder gru --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-073802_pointmaze_gru_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py -e pointmaze_trans2_F --env pointmaze --encoder trans --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2 -T g0128-230116_pointmaze_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env pointmaze --encoder tree_lstm --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-073826_pointmaze_tree_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env pointmaze --encoder goal --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-073910_pointmaze_goal_F --fix --num_evals 128 -b 1 --test_muls 1024

# fast-flow
python train_gstl_v1.py --env pointmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-073754_pointmaze_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _V1 -V 1

# LTLDoG
python train_gstl_v1.py --env pointmaze --encoder goal --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-073910_pointmaze_goal_F --cls_guidance -C g0129-114304_score_pointmaze_gnn --guidance_before 10 --guidance_lr 0.2 --guidance_steps 1 --smoothing_factor 500 --guidance_scale 1.0 --fix --suffix ltl --num_evals 128 -b 1 --test_muls 1024

# diffusion (g0128-194756_pointmaze_gnn_diffusion)
python train_gstl_v1.py -e pointmaze_gnn_diffusion --env pointmaze --encoder gnn --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-194756_pointmaze_gnn_diffusion --fix --num_evals 128 -b 1 --test_muls 1024

python train_gstl_v1.py --env pointmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --skip_first_eval --clip_max 160000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-073754_pointmaze_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix __viz16_23 --mini --max_viz 16 -V 23


#######################
# env-3 antmaze
# Different archs: Ours/GRU/Transformer/TreeLSTM/Goal-conditioned
python train_gstl_v1.py --env antmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-164357_antmaze_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env antmaze --encoder gru --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto -T g0128-164414_antmaze_gru_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py -e antmaze_trans2_F --env antmaze --encoder trans --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashimoto --nlayers 2 -T g0128-230142_antmaze_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env antmaze --encoder tree_lstm --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-074038_antmaze_tree_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env antmaze --encoder goal --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-074048_antmaze_goal_F --fix --num_evals 128 -b 1 --test_muls 1024

# fast-flow
python train_gstl_v1.py --env antmaze --encoder gnn --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-164357_antmaze_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _V1 -V 1

# LTLDoG
python train_gstl_v1.py --env antmaze --encoder goal --flow --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-074048_antmaze_goal_F --cls_guidance -C g0129-160750_score_antmaze_gnn --guidance_before 10 --guidance_lr 0.2 --guidance_steps 1 --smoothing_factor 500 --guidance_scale 1.0 --fix --suffix ltl --num_evals 128 -b 1 --test_muls 1024

# diffusion (g0128-200948_antmaze_gnn_diffusion)
python train_gstl_v1.py -e antmaze_gnn_diffusion --env antmaze --encoder gnn --horizon 512 --normalize --stat_decay 2.5 --tconv_dim 64 --dim_mults 1 2 4 16 --partial_traj --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-200948_antmaze_gnn_diffusion --fix --num_evals 128 -b 1 --test_muls 1024


#######################
# env-4 panda
# Different archs: Ours/GRU/Transformer/TreeLSTM/Goal-conditioned
python train_gstl_v1.py --env panda --encoder gnn --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-084928_panda_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env panda --encoder gru --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --bfs_encoding -T g0128-162749_panda_gru_bfs_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env panda --encoder trans --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --nlayers 2 --hashimoto -T g0129-203130_panda_trans2_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env panda --encoder tree_lstm --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 --hashi_gnn -T g0128-084943_panda_tree_F --fix --num_evals 128 -b 1 --test_muls 1024
python train_gstl_v1.py --env panda --encoder goal --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-084948_panda_goal_F --fix --num_evals 128 -b 1 --test_muls 1024

# fast-flow
python train_gstl_v1.py --env panda --encoder gnn --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-084928_panda_gnn_F --fix --num_evals 128 -b 1 --test_muls 1024 --suffix _V1 -V 1

# CTG
python train_gstl_v1.py --env panda --encoder goal --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-084948_panda_goal_F --guidance --guidance_before 10 --guidance_lr 0.2 --guidance_steps 3 --smoothing_factor 500 --guidance_scale 1.0 --fix --suffix ctg --num_evals 128 -b 1 --test_muls 1024

# LTLDoG
python train_gstl_v1.py --env panda --encoder goal --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-084948_panda_goal_F -C g0129-203314_score_panda_gnn --cls_guidance --guidance_before 10 --guidance_lr 0.2 --guidance_steps 3 --smoothing_factor 500 --guidance_scale 1.0 --fix --suffix ltl --num_evals 128 -b 1 --test_muls 1024

# CEM
python train_gstl_v1.py --env panda --encoder gnn --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-084928_panda_gnn_F --cem --fix --suffix cem --num_evals 128 -b 1

# Grad (iters=50/150)
python train_gstl_v1.py --env panda --encoder gnn --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-084928_panda_gnn_F --trajopt --fix --suffix tj_50 --num_evals 128 -b 1 --trajopt_niters 50 --trajopt_lr 3e-2 
python train_gstl_v1.py --env panda --encoder gnn --flow --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-084928_panda_gnn_F --trajopt --fix --suffix tj_150 --num_evals 128 -b 1 --trajopt_niters 150 --trajopt_lr 3e-2 

# diffusion (g0128-204325_panda_gnn_diffusion)
python train_gstl_v1.py -e panda_gnn_diffusion --env panda --encoder gnn --tconv_dim 64 --skip_first_eval --clip_max 100000 --max_sol_clip 2 --type_ratios 1 3 2 4 -T g0128-204325_panda_gnn_diffusion --fix --num_evals 128 -b 1 --test_muls 1024
