
## 1.Prerequisities
### 1.1 Ubuntu20.04 and ROS noetic
### 1.2 OpenCV 4.2
### 1.3 Eigen3
### 1.4 Casadi
### 1.4 Ipopt
The solver we use is MA57,[mpc](../src/ocp_planner/src/mpc.cc) which requires academic verification to obtain.
### 1.5 Conda Env
simple use
```bash
conda env create -f environment.yml
```

### 1.5 pybind11(Optional but recommended)
We implemented two versions: one as a dynamic library built with pybind11, and the other based on a ROS service."

```bash
cd ~/RL_MPC/src/ocp_planner/
mkdir extern && cd extern
git clone https://github.com/pybind/pybind11.git
```


## 2.Install


```bash
git clone https://github.com/TIB-K330/HALO.git
cd HALO
.build_cp.sh

```
## 3.Usage
- train

```bash
cd ~/HALO
source ./devel/setup.bash
conda activate mpc_rl && export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
cd drl_moudle
python train_ppo.py

```

- eval

```bash
cd ~/HALO/drl_moudle/drl_moudle
python eval_ppo.py

```

## 4.Supplement
Our MPC solver uses **Ipopt** and **FORCES PRO**. You can implement it with your own MPC as long as the output form is the same as ours.



