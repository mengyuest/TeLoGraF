# TeLoGraF: Temporal Logic Planning via Graph-encoded Flow Matching

[![Conference](https://img.shields.io/badge/ICML2025-Accepted-success)](https://icml.cc/Conferences/2025/)
[![Conference](https://img.shields.io/badge/Arxiv-Paper-success?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2504.03015)



[<ins>Yue Meng</ins>](https://mengyuest.github.io/), [<ins>Chuchu Fan</ins>](https://chuchu.mit.edu/)

[<ins>Reliable Autonomous Systems Lab @ MIT (REALM)</ins>](https://aeroastro.mit.edu/realm/)

[[ICML2025-SlidesLive]](https://recorder-v3.slideslive.com/?share=101566&s=d301d764-3864-46fc-9df7-7f64904b6b9d)




> A graph-based flow-matching framework that learns to solve general Signal Temporal Logic (STL) tasks across complex systems from a large dataset of specifications and demonstrations.

![Figure](figures/fig0_arch.png)
<p align="center"><em>Graph-encoding flow matching framework.</em></p>


![Figure](figures/fig1_syntax.png)
<p align="center"><em>TeLoGraF can handle general STL specifications.</em></p>


| ![](figures/env1_linear.png) | ![](figures/env2_nonlinear.png) | ![](figures/env3_pointmaze.png) | ![](figures/env4_antmaze.png) | ![](figures/env5_panda.png) |
|:----------------------------:|:-------------------------------:|:-------------------------------:|:-----------------------------:|:---------------------------:|
| Linear                      | Nonlinear                       | PointMaze                       | AntMaze                       | Panda                       |


<p align="center"><em>TeLoGraF solves for a wide range of task environments.</em></p>

![Figure](figures/fig2_results.png)
<p align="center"><em>Strong results in STL success rate and runtime efficiency.</em></p>

## Prerequisite
> Ubuntu 22.04 / 24.04 with GPU (Nvidia V100 / L40S / RTX4090); Python=3.10.16
1. Install required packages.
```bash
conda create -y -n telograf python=3.10 && conda activate telograf
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 matplotlib pyg imageio -c pytorch -c nvidia -c pyg
pip install networkx gurobipy cma pandas einops
sudo apt-get update && sudo apt-get install -y libegl1-mesa libgles2-mesa
pip install "stable-baselines3[extra]"
pip install "gymnasium[all]"
pip install gymnasium-robotics
pip install "minari[all]"
pip install pybullet pytorch_kinematics
```
2. Specify the experiment directory by using the environment variable `export MY_EXPS_DIR=/foo/bar` so that the experiment dir is `/foo/bar/exps_gstl` (if you don't specify it, the default directory is `../exps_gstl`). Recommend to write the `export ...` command in the `.bashrc` file.
3. Download the datasets from [Google Drive](https://drive.google.com/drive/folders/1SZsTEiJluv9Bm8GPMvpmRPzRc_4bFtos?usp=sharing) to the experiment directory.
4. Download the pretrained model checkpoints from [Google Drive](https://drive.google.com/drive/folders/1DYqgMYrg0zfkkhtUfVhLlmOmXXu0pQki?usp=sharing), and unzip them to the experiment directory.
5. The expected file structure will be (if do not specify `MY_EXPS_DIR`):
```bash
exps_gstl
└──── data_0_simple
│     │     data.npz
│     │     ...
│
└──── g0128xxxxxxxx
      │     log-xxxx-xxxxxx.txt
      │     ...
      └──── models
            │      model_last.ckpt
            │      ...

project
└──── figures
│
└──── code
      │     train_gstl_v1.py
      │     ... 
```

## Usage
All the operations below assume you are in the `code` directory.

### Testing
1. Run the commands in `run_icml2025_test.sh` to get the main results.
2. Run the commands in `run_icml2025_test_ood.sh` to get the OOD test results.
3. Run `python plot_figures.py` to get the figures shown in the paper. The figures will be saved in the experiment directory.

### Training
1. Run the commands in `run_icml2025_train.sh` to train the required neural networks. It might take 5-24 hours depending on the GPU configuration. The models will be saved in the experiment directory.
2. To run the test with your trained models, replace the `-T` or `-C` flag values in the `run_icml2025_test*` files with your own model directory.


## Reference
```bibtex
@article{meng2025telograf,
  title={TeLoGraF: Temporal Logic Planning via Graph-encoded Flow Matching},
  author={Meng, Yue and Fan, Chuchu},
  journal={International Conference on Machine Learning},
  year={2025}
}
```