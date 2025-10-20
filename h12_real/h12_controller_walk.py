"""
H12 Controller - Walk (Real hardware wrapper)
Mirrors the MuJoCo H12_Controller_Walk API but adapted to real hardware.
Loads the `walk` policy from `h1_2.yaml` and exposes methods to update
state and compute target leg positions.
"""
import time
import yaml
import torch
import numpy as np
import collections


def load_config(path="h1_2.yaml"):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


class H12_Controller_Walk:
    def __init__(self, config_path="h1_2.yaml"):
        cfg = load_config(config_path)
        shared = cfg.get('shared_params', {})
        walk = cfg.get('policies', {}).get('walk', {})
        self.config = {**shared, **walk}

        # policy
        self.policy = torch.jit.load(self.config['policy_path'])

        # state
        self.action = np.zeros(self.config.get('num_actions', 12), dtype=np.float32)
        self.target_dof_legs_pos = np.array(self.config.get('default_angles_legs', [0.0]*12), dtype=np.float32)

        # command state for walking
        self.cmd_vel = np.zeros(3, dtype=np.float32)
        self.height_cmd = float(self.config.get('height_cmd', 1.05))

        # observation settings
        self.single_obs_dim = int(self.config.get('single_obs_dim', 76))
        self.obs_history_len = int(self.config.get('obs_history_len', 1))
        self.obs_history = collections.deque(maxlen=self.obs_history_len)
        for _ in range(self.obs_history_len):
            self.obs_history.append(np.zeros(self.single_obs_dim, dtype=np.float32))

        # internal buffers
        self.qj = np.zeros(self.config.get('num_dofs', 27), dtype=np.float32)
        self.dqj = np.zeros(self.config.get('num_dofs', 27), dtype=np.float32)

        self.counter = 0

    def update_state_from_low_state(self, low_state):
        all_idxs = self.config.get('leg_joint2motor_idx', []) + self.config.get('arm_waist_joint2motor_idx', [])
        for i, motor_idx in enumerate(all_idxs):
            self.qj[i] = low_state.motor_state[motor_idx].q
            self.dqj[i] = low_state.motor_state[motor_idx].dq

    def construct_observation(self, quat, ang_vel):
        # Phase calculation similar to MuJoCo to provide gait phase
        period = 0.8
        phase = (self.counter * float(self.config.get('control_dt', 0.02))) % period / period
        sin_cos_phase = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], dtype=np.float32)

        # scaled values
        omega_scaled = ang_vel * float(self.config.get('ang_vel_scale', 0.25))
        gravity_orientation = quat
        cmd_scaled = self.cmd_vel * np.array(self.config.get('cmd_scale', [1.0, 1.0, 1.0]), dtype=np.float32)
        qj_scaled = (self.qj[:12] - np.array(self.config.get('default_angles_legs', [0.0]*12))) * float(self.config.get('dof_pos_scale', 1.0))
        dqj_scaled = self.dqj[:12] * float(self.config.get('dof_vel_scale', 0.05))

        obs = np.concatenate([
            omega_scaled, gravity_orientation, cmd_scaled, qj_scaled, dqj_scaled,
            self.action, sin_cos_phase
        ]).astype(np.float32)

        return obs

    def infer_policy(self):
        single = self.construct_observation(np.zeros(3), np.zeros(3))
        self.obs_history.append(single)
        obs = np.concatenate(list(self.obs_history)).astype(np.float32)
        with torch.no_grad():
            out = self.policy(torch.from_numpy(obs).unsqueeze(0))
        self.action = out.detach().cpu().numpy().squeeze()

        scaled_action = self.action * float(self.config.get('action_scale', 0.25))
        lowers = np.array(self.config.get('legs_motor_pos_lower_limit_list', [-10.0]*12), dtype=np.float32)
        uppers = np.array(self.config.get('legs_motor_pos_upper_limit_list', [10.0]*12), dtype=np.float32)
        clipped = np.clip(scaled_action, lowers, uppers)
        self.target_dof_legs_pos = clipped + np.array(self.config.get('default_angles_legs', [0.0]*12), dtype=np.float32)

    def get_target_positions(self):
        return self.target_dof_legs_pos.copy()

