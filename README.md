# h12_loco_manipulation

# 0. Prerequisites

- Ubuntu 22.04 Operating System
- IsaacGym Preview 4.0
  - NVIDIA GPU (RTX 4070 )
  - NVIDIA GPU Driver (Driver Version: 570.124.06     CUDA Version: 12.8 )



# 1. Installation Guide

## System Requirements

- **Operating System**: Recommended Ubuntu 18.04 or later  
- **GPU**: Nvidia GPU  
- **Driver Version**: Recommended version 525 or later  


### 1.1 Create a New Environment

Use the following command to create a virtual environment:

```bash
conda create -n unitree_rl python=3.8
```

### 1.2 Activate the Virtual Environment

```bash
conda activate unitree_rl
```

### 1.3 Installing Dependencies 
#### 1.3.1 Install PyTorch

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```


#### 1.3.2 Install Isaac Gym

Isaac Gym is a rigid body simulation and training framework provided by Nvidia.

##### Download

Download [Isaac Gym](https://developer.nvidia.com/isaac-gym) from Nvidia’s official website.

##### Install

After extracting the package, navigate to the `isaacgym/python` folder and install it using the following commands:

```bash
cd isaacgym/python
pip install -e .
```

#### 1.3.3 Install rsl_rl


##### Download

Clone the repository using Git:

```bash
git clone https://github.com/leggedrobotics/rsl_rl.git
```

##### Switch Branch

Switch to the v1.0.2 branch:

```bash
cd rsl_rl
git checkout v1.0.2
```

##### Install

```bash
pip install -e .
```


# 2. Install h12_loco_manipulation


#### Download

Clone the repository using Git:

```bash
git clone https://github.com/correlllab/h12_loco_manipulation.git
```

#### Install

Navigate to the directory and install it:

```bash
cd h12_loco_manipulation
pip install -r requirements
```

### Install unitree_sdk2py (Optional ~ For Real Deployment)

`unitree_sdk2py` is a library used for communication with real robots. If you need to deploy the trained model on a physical robot, install this library.

#### Download

Clone the repository using Git:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
```

#### Install

Navigate to the directory and install it:

```bash
cd unitree_sdk2_python
pip install -e .
```

Then you can follow the README.md in each sub-repostory to install all three parts or just one of them.
<!-- 
## 📝 TODO List

- \[\]
- \[\]
- \[\]
- \[\] -->

## 📄 License

## Acknowledgements
- [OpenHomie](https://github.com/InternRobotics/OpenHomie/tree/main): We use OpenHomie library as our codebase.
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl): We use rsl_rl library to train the control policies for legged robots.
- [Legged_gym](https://github.com/leggedrobotics/rsl_rl): We use legged_gym library to train the control policies for legged robots.
- [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python) : We use Unitree SDK2 Python library from Unitree to control the real robot.
