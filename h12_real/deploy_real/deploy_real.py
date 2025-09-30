# from legged_gym import LEGGED_GYM_ROOT_DIR

import os
import pickle
import pygame
import matplotlib.pyplot as plt
from typing import Union
import numpy as np
import time
import torch

import collections


from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config

######################################################################
## Input & Plotting Configuration
######################################################################

# Keyboard state tracking for pygame
key_states = {
    "r": False, "f": False, "x": False,
}

# Joint names for plotting labels
JOINT_NAMES_PLOT = [
    "L_hip_yaw", "L_hip_pitch", "L_hip_roll", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_yaw", "R_hip_pitch", "R_hip_roll", "R_knee", "R_ankle_pitch", "R_ankle_roll"
]

######################################################################
## Utility Functions
######################################################################

def handle_input(cmd, delta=0.0005):
    """Handles keyboard input for adjusting commands."""
    global key_states
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
            key_name = pygame.key.name(event.key)
            if key_name in key_states:
                key_states[key_name] = event.type == pygame.KEYDOWN

    if key_states["r"]:
        cmd["height"] = min(cmd["height"] + delta, 1.03)
    if key_states["f"]:
        cmd["height"] = max(cmd["height"] - delta, 0.70)
    if key_states["x"]:
        cmd = {"x": 0.0, "y": 0.0, "yaw": 0.0, "height": 1.0}
    return cmd

def plot_qpos_vs_action(t, qpos_hist, target_dof_hist, joint_names, save_path):
    """Plots measured joint positions against commanded positions."""
    n_joints = len(joint_names)
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 2.5 * n_joints), sharex=True)
    if n_joints == 1: axes = [axes]
    for i, name in enumerate(joint_names):
        axes[i].plot(t, qpos_hist[:, i], label="qpos (measured)", color="blue")
        axes[i].plot(t, target_dof_hist[:, i], label="target_dof_pos (command)", color="green", linestyle="--")
        axes[i].set_ylabel(name)
        axes[i].grid(True)
        if i == 0: axes[i].set_title("Measured Joint Position vs. Commanded Action")
        axes[i].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved overlay plot to {save_path}")
    plt.close(fig)

def plot_dqpos(t, dqpos_hist, joint_names, save_path):
    """Plots measured joint velocities."""
    # (Implementation is identical to your original code)
    n_joints = len(joint_names)
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 2.5 * n_joints), sharex=True)
    if n_joints == 1: axes = [axes]
    for i, name in enumerate(joint_names):
        axes[i].plot(t, dqpos_hist[:, i], label="dqpos (measured)", color="orange")
        axes[i].set_ylabel(name)
        axes[i].grid(True)
        if i == 0: axes[i].set_title("Measured Joint Velocity")
        axes[i].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved velocity plot to {save_path}")
    plt.close(fig)

######################################################################
## Main Controller Class
######################################################################
class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)

        # Initializing process variables
        self.qj = np.zeros(config.num_dofs, dtype=np.float32)
        self.dqj = np.zeros(config.num_dofs, dtype=np.float32)

        self.action = np.zeros(config.num_actions, dtype=np.float32)

        # RL observation vector (full size with history)
        self.target_dof_pos = config.default_angles.copy()
        
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0.0, 0.0])
        self.height_cmd = np.array(1.03)
        self.counter = 0

        # Histories for data logging
        self.qpos_hist, self.dqpos_hist, self.target_dof_hist, self.t_hist = [], [], [], []
        self.start_time = time.time()

        self.single_obs_dim = (config.num_obs // config.obs_history_len)
        self.obs_history = collections.deque(maxlen=config.obs_history_len)

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        print("waiting")
        self.wait_for_low_state()
        print("wait complete")
        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("▶️ Entering zero torque state. Press 'START' on controller to proceed...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("▶️ Moving to default position...")
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)

        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        print("✅ Reached default position.")

    def default_pos_state(self):
        print("▶️ Holding default position. Press 'A' on controller to start RL policy...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        print("✅ RL Policy Engaged!")

    def run(self):
        self.counter += 1
        t_start = time.time()  # <-- ADD THIS LINE

        input_cmd = {
            "x": self.cmd[0], "y": self.cmd[1], "yaw": self.cmd[2], "height": self.height_cmd
        }
        input_cmd = handle_input(input_cmd)
        
        # Update internal commands from the returned dictionary
        self.cmd = np.array([input_cmd["x"], input_cmd["y"], input_cmd["yaw"]])
        self.height_cmd = input_cmd["height"]
        

        print("current height: ",self.height_cmd)


        full_default_angles = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)

        # Get the current joint position and velocity
        all_motor_indices = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx

        for i, motor_idx in enumerate(all_motor_indices):
            # This populates self.qj[0:20] and self.dqj[0:20] (assuming 20 total DOFs)
            self.qj[i] = self.low_state.motor_state[motor_idx].q
            self.dqj[i] = self.low_state.motor_state[motor_idx].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)

        num_dofs = self.config.num_dofs   # 27 (qj/dqj size)
        num_actions = self.config.num_actions # 12 (Policy action size, usually legs)
        
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        
        qj_obs = (qj_obs - full_default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale


        # Single observation size = 3 + 1 + 3 + 3 + 27 + 27 + 12 = 76 (assuming 12 actions)
        single_obs_dim = 3 + 1 + 3 + 3 + num_dofs + num_dofs + num_actions
        single_obs = np.zeros(single_obs_dim, dtype=np.float32)
        
        current_idx = 0
        single_obs[current_idx : current_idx + 3] = self.cmd
        current_idx += 3
        
        single_obs[current_idx] = self.height_cmd
        current_idx += 1

        # 2. omega_scaled (3 values)
        single_obs[current_idx : current_idx + 3] = ang_vel
        current_idx += 3

        # 3. grav_orientation (3 values)
        single_obs[current_idx : current_idx + 3] = gravity_orientation
        current_idx += 3
        
        # 4. qj_scaled (27 values)
        single_obs[current_idx : current_idx + num_dofs] = qj_obs
        current_idx += num_dofs
        
        # 5. dqj_scaled (27 values)
        single_obs[current_idx : current_idx + num_dofs] = dqj_obs
        current_idx += num_dofs


         # 6. last_action (12 values)
        single_obs[current_idx : current_idx + num_actions] = self.action
        # current_idx += num_actions # Not strictly needed if this is the end
        
        # --- 4. HISTORY & STACK ---
        self.obs_history.append(single_obs.copy())

        # Construct full observation with history (self.obs)
        for i, hist_obs in enumerate(self.obs_history):
            start_idx = i * single_obs_dim
            self.obs[start_idx : start_idx + single_obs_dim] = hist_obs

        # --- 5. POLICY INFERENCE ---
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().cpu().numpy().squeeze()

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()

        scaled_action = self.action * self.config.action_scale

        clipped_action = np.clip(
                            scaled_action,
                            np.array(self.config.legs_motor_pos_lower_limit_list),
                            np.array(self.config.legs_motor_pos_upper_limit_list))
        
        target_dof_pos = self.config.default_angles + clipped_action



        # --- DATA LOGGING ADDITIONS ---
        current_time = time.time() - self.start_time
        self.t_hist.append(current_time)
        self.qpos_hist.append(self.qj.copy()) # Full qpos (27 DOFs)
        self.dqpos_hist.append(self.dqj.copy()) # Full dqpos (27 DOFs)
        full_target_dof = np.concatenate((target_dof_pos, self.config.arm_waist_target), axis=0)
        self.target_dof_hist.append(full_target_dof) 
        # -----------------------------


        # if self.counter <= 5:
        #     print(f"[{self.counter}] target dof: {target_dof_pos}")
            
        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        if self.counter <= 2:
            print(f"[{self.counter}] obs_history: {self.obs_history}")
            
        self.send_cmd(self.low_cmd)
      
        t_elapsed = time.time() - t_start
        time_to_sleep = self.config.control_dt - t_elapsed
        if time_to_sleep > 0:
             time.sleep(time_to_sleep)

        # Now, measure the time of the ENTIRE cycle, including the sleep
        t_full_cycle = time.time() - t_start
        
        # Print the full cycle time and the corresponding fixed frequency (50 Hz)
        print(f"Full Cycle Time: {t_full_cycle:.5f}s (Target: {self.config.control_dt:.3f}s), Frequency: {1.0/t_full_cycle:.1f} Hz")


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="h1_2.yaml")
    args = parser.parse_args()

    config_path = f"/home/niraj/isaac_projects/h12_loco_manipulation/h12_real/deploy_real/configs/{args.config}"
    config = Config(config_path)

    print("Config:", config.action_scale, config.cmd_scale, config.dof_pos_scale, config.dof_vel_scale, config.ang_vel_scale)
    print("Policy:", config.policy_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    pygame.init() 
    pygame.display.set_mode((300, 100)) # Small window for pygame input

    controller.zero_torque_state()

    controller.move_to_default_pos()

    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")


    # ----------------------------------------------------------------------------------
    # --- POST-EXECUTION PLOTTING ---
    # ----------------------------------------------------------------------------------

        # Ensure logs directory exists
    log_dir = "logs/real"
    os.makedirs(log_dir, exist_ok=True)
    
    qpos_hist_arr = np.array(controller.qpos_hist)
    dqpos_hist_arr = np.array(controller.dqpos_hist)
    target_dof_hist_arr = np.array(controller.target_dof_hist)
    t_hist_arr = np.array(controller.t_hist)

    if not t_hist_arr.size:
        print("No data logged, skipping plots.")
        sys.exit() # Exit after cleanup and message if no data

    # Plot qpos vs. target_dof_pos
    plot_qpos_vs_action(
        t_hist_arr, 
        qpos_hist_arr, 
        target_dof_hist_arr, 
        JOINT_NAMES_PLOT, 
        os.path.join(log_dir, "qpos_vs_target.png")
    )
    
    # Plot dqpos
    plot_dqpos(
        t_hist_arr, 
        dqpos_hist_arr, 
        JOINT_NAMES_PLOT, 
        os.path.join(log_dir, "dqpos.png")
    )