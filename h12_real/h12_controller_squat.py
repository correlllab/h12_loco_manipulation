"""
H12 Controller - Squat (Real hardware wrapper)
This file provides an API similar to the MuJoCo `H12_Controller_Squat` but
adapted for the real robot (DDS low_state). It loads the `squat` policy from
the split `h1_2.yaml` and exposes methods to update state and compute
target leg positions which can be written to the Unitree low_cmd message.
"""
import time
import collections
import yaml
import torch
import numpy as np


def load_config(path="h1_2.yaml"):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


class H12_Controller_Squat:
    """Squat policy wrapper for real hardware.

    Public attributes & methods mirror the MuJoCo wrapper so deploy scripts
    can switch between real/sim easily.
    """

    def __init__(self, config_path="h1_2.yaml"):
        cfg = load_config(config_path)
        shared = cfg.get('shared_params', {})
        squat = cfg.get('policies', {}).get('squat', {})
        # merged config
        self.config = {**shared, **squat}

        # policy
        self.policy = torch.jit.load(self.config['policy_path'])

        # state
        self.action = np.zeros(self.config.get('num_actions', 12), dtype=np.float32)
        self.target_dof_legs_pos = np.array(self.config.get('default_angles_legs', [0.0]*12), dtype=np.float32)

        # commands
        self.cmd = np.array(self.config.get('cmd_init', [0.0, 0.0, 0.0]), dtype=np.float32)
        self.height_cmd = float(self.config.get('height_cmd', 1.05))

        # observation history
        self.single_obs_dim = int(self.config.get('single_obs_dim', 76))
        self.obs_history_len = int(self.config.get('obs_history_len', 6))
        self.obs_history = collections.deque(maxlen=self.obs_history_len)
        for _ in range(self.obs_history_len):
            self.obs_history.append(np.zeros(self.single_obs_dim, dtype=np.float32))

        # internal buffers
        self.qj = np.zeros(self.config.get('num_dofs', 27), dtype=np.float32)
        self.dqj = np.zeros(self.config.get('num_dofs', 27), dtype=np.float32)

        self.counter = 0

    def update_state_from_low_state(self, low_state):
        """Populate qj and dqj from Unitree LowState message.
        Expects low_state.motor_state list where each has `.q` and `.dq`.
        """
        all_idxs = self.config.get('leg_joint2motor_idx', []) + self.config.get('arm_waist_joint2motor_idx', [])
        for i, motor_idx in enumerate(all_idxs):
            self.qj[i] = low_state.motor_state[motor_idx].q
            self.dqj[i] = low_state.motor_state[motor_idx].dq

    def construct_observation(self, quat, ang_vel):
        """Construct the single-frame observation used by the squat policy.
        This follows the same layout as the MuJoCo version so policies remain
        compatible.
        """
        # default angles combine legs + arms if available
        default_legs = np.array(self.config.get('default_angles_legs', [0.0]*12), dtype=np.float32)
        default_arms = np.array(self.config.get('default_angles_arms', []), dtype=np.float32)
        full_default = np.concatenate((default_legs, default_arms)) if default_arms.size else default_legs

        # scale qj and dqj
        qj_scaled = (self.qj[:len(full_default)] - full_default) * float(self.config.get('dof_pos_scale', 1.0))
        dqj_scaled = self.dqj[:len(full_default)] * float(self.config.get('dof_vel_scale', 0.05))

        gravity_orientation = quat  # assume caller provides gravity vector if needed
        omega_scaled = ang_vel * float(self.config.get('ang_vel_scale', 0.25))

        cmd_array = np.array(self.cmd, dtype=np.float32) * np.array(self.config.get('cmd_scale', [1.0, 1.0, 1.0]), dtype=np.float32)

        single = np.zeros(self.single_obs_dim, dtype=np.float32)
        idx = 0
        single[idx:idx+3] = cmd_array; idx += 3
        single[idx] = self.height_cmd; idx += 1
        single[idx:idx+3] = omega_scaled; idx += 3
        single[idx:idx+3] = gravity_orientation; idx += 3
        n = len(qj_scaled)
        single[idx:idx+n] = qj_scaled; idx += n
        single[idx:idx+n] = dqj_scaled; idx += n
        single[idx:idx+12] = self.action

        return single

    def infer_policy(self):
        """Run the squat policy and update target leg positions."""
        single = self.construct_observation(np.zeros(3), np.zeros(3))
        self.obs_history.append(single)
        obs = np.concatenate(list(self.obs_history)).astype(np.float32)
        with torch.no_grad():
            out = self.policy(torch.from_numpy(obs).unsqueeze(0))
        self.action = out.detach().cpu().numpy().squeeze()

        scaled = self.action * float(self.config.get('action_scale', 0.25))
        lowers = np.array(self.config.get('legs_motor_pos_lower_limit_list', [-10.0]*12), dtype=np.float32)
        uppers = np.array(self.config.get('legs_motor_pos_upper_limit_list', [10.0]*12), dtype=np.float32)
        clipped = np.clip(scaled, lowers, uppers)
        self.target_dof_legs_pos = clipped + np.array(self.config.get('default_angles_legs', [0.0]*12), dtype=np.float32)

    def get_target_positions(self):
        """Return current target leg positions (12 values) to be written to low_cmd."""
        return self.target_dof_legs_pos.copy()

