
### Sim2Sim (Mujoco)

Run Sim2Sim in the Mujoco simulator:

## Installation

First install MuJoCo:
```bash
pip install mujoco==3.2.3
```

#### Parameter Description
- `config_name`: Configuration file; default search path is `h12_mujoco/h1_2.yaml`.

#### Example: Running H1_2


```bash
cd ~/h12_loco_manipulation/h12_mujoco
python mujoco_deploy_h12.py
```

#### Replace Network Model

Once you run play.py from the HomieRL, the exported model is located at `{LEGGED_GYM_ROOT_DIR}/logs/exported/policies/policy.pt`. Update the `policy_path` in the YAML configuration file accordingly.

#### Simulation Results

## Acknowledgments

This repository is built upon the support and contributions of the following open-source projects:

- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
---
