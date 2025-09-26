import os
import pickle

import sys
import time
import collections
import logging
import yaml
import torch
import numpy as np
import mujoco
import mujoco.viewer

import pygame

import os
import pickle
######################################################################

# Track key states manually
key_states = {
    "w": False, "s": False, "a": False, "d": False,
    "q": False, "e": False, "r": False, "f": False, "x":False,
}


# Histories for logging
qpos_hist, dqpos_hist, target_dof_hist, t_hist = [], [], [], []

# Joint names (12 dof, adapt if needed)
joint_names = [
    "L_hip_yaw", "L_hip_pitch", "L_hip_roll", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_yaw", "R_hip_pitch", "R_hip_roll", "R_knee", "R_ankle_pitch", "R_ankle_roll"
]


# ----------------------------
# Plotting
# ----------------------------

import matplotlib.pyplot as plt

def plot_qpos_vs_action(t, qpos_hist, target_dof_hist, joint_names, save_path="logs/qpos_vs_action.png"):
    n_joints = len(joint_names)
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 2.5*n_joints), sharex=True)

    if n_joints == 1:
        axes = [axes]

    for i, name in enumerate(joint_names):
        axes[i].plot(t, qpos_hist[:, i], label="qpos", color="blue")
        axes[i].plot(t, target_dof_hist[:, i], label="scaled_action", color="green", linestyle="--")
        axes[i].set_ylabel(name)
        axes[i].grid(True)
        if i == 0:
            axes[i].set_title("qpos vs scaled_action")
        if i == n_joints - 1:
            axes[i].set_xlabel("Time [s]")
        axes[i].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved overlay plot (qpos vs action) to {save_path}")
    plt.close(fig)


def plot_dqpos(t, dqpos_hist, joint_names, save_path="logs/dqpos.png"):
    n_joints = len(joint_names)
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 2.5*n_joints), sharex=True)

    if n_joints == 1:
        axes = [axes]

    for i, name in enumerate(joint_names):
        axes[i].plot(t, dqpos_hist[:, i], label="dqpos", color="orange")
        axes[i].set_ylabel(name)
        axes[i].grid(True)
        if i == 0:
            axes[i].set_title("dqpos")
        if i == n_joints - 1:
            axes[i].set_xlabel("Time [s]")
        axes[i].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved dqpos plot to {save_path}")
    plt.close(fig)



def handle_input(cmd, delta=0.0008):
    global key_states

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            key_name = pygame.key.name(event.key)
            if key_name in key_states:
                key_states[key_name] = True
        elif event.type == pygame.KEYUP:
            key_name = pygame.key.name(event.key)
            if key_name in key_states:
                key_states[key_name] = False

    # # Forward/Backward (cmd[0])
    # if key_states["w"]:
    #     cmd["x"] = min(cmd["x"] + delta, 1.5)
    # if key_states["s"]:
    #     cmd["x"] = max(cmd["x"] - delta, -1.5)

    # # Left/Right (cmd[1])
    # if key_states["d"]:
    #     cmd["y"] = min(cmd["y"] + delta, 1.0)
    # if key_states["a"]:
    #     cmd["y"] = max(cmd["y"] - delta, -1.0)

    # # Yaw rate (cmd[2])
    # if key_states["q"]:
    #     cmd["yaw"] = min(cmd["yaw"] + delta, 0.5)
    # if key_states["e"]:
    #     cmd["yaw"] = max(cmd["yaw"] - delta, -0.5)

    # Height
    if key_states["r"]:
        cmd["height"] = min(cmd["height"] + delta, 1.0)
    if key_states["f"]:
        cmd["height"] = max(cmd["height"] - delta, 0.65)


    if key_states["x"]:
        cmd = {
        "x": 0.0,
        "y": 0.0,
        "yaw": 0.0,
        "height": 1.00,
    }
    return cmd

######################################################################

def load_config(config_path):
    """Load and process the YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Process paths with LEGGED_GYM_ROOT_DIR
    for path_key in ['policy_path', 'xml_path']:
        config[path_key] = config[path_key]

    # Convert lists to numpy arrays where needed
    array_keys = ['kps_legs', 'kds_legs', 'default_angles_legs','kps_arms', 'kds_arms', 'default_angles_arms', 'cmd_scale', 'cmd_init', 'legs_motor_pos_lower_limit_list', 'legs_motor_pos_upper_limit_list']
    for key in array_keys:
        config[key] = np.array(config[key], dtype=np.float32)

    return config

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q"""
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    q_conj = np.array([w, -x, -y, -z])

    return np.array([
        v[0] * (q_conj[0]**2 + q_conj[1]**2 - q_conj[2]**2 - q_conj[3]**2) +
        v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3]) +
        v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),

        v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3]) +
        v[1] * (q_conj[0]**2 - q_conj[1]**2 + q_conj[2]**2 - q_conj[3]**2) +
        v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),

        v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2]) +
        v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1]) +
        v[2] * (q_conj[0]**2 - q_conj[1]**2 - q_conj[2]**2 + q_conj[3]**2)
    ])

def get_gravity_orientation(quat):
    """Get gravity vector in body frame"""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)


def compute_observation(d, config, action, cmd, height_cmd, n_joints):
    """Compute the observation vector from current state"""
    # Get state from MuJoCo
    qj = d.qpos[7:7+n_joints].copy()
    dqj = d.qvel[6:6+n_joints].copy()
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()

    # Handle default angles padding
    default_joints = np.concatenate((config['default_angles_legs'], config['default_angles_arms']))

    # Scale the values
    qj_scaled = (qj - default_joints) * config['dof_pos_scale']
    dqj_scaled = dqj * config['dof_vel_scale']
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega * config['ang_vel_scale']

    if isinstance(cmd, dict):
        cmd_array = np.array([cmd["x"], cmd["y"], cmd["yaw"]])
    else:
        cmd_array = np.array(cmd)

    # Calculate single observation dimension
    single_obs_dim = 3 + 1 + 3 + 3 + n_joints + n_joints + 12

    # Create single observation
    single_obs = np.zeros(single_obs_dim, dtype=np.float32)

    single_obs[0:3] = cmd_array* config['cmd_scale']
    single_obs[3:4] = np.array([height_cmd])

   # print("CMD:", single_obs[0:4])

    single_obs[4:7] = omega_scaled
    single_obs[7:10] = gravity_orientation
    single_obs[10:10+n_joints] = qj_scaled
    single_obs[10+n_joints:10+2*n_joints] = dqj_scaled
    single_obs[10+2*n_joints:10+2*n_joints+12] = action



    qpos_hist.append(qj.copy())
    dqpos_hist.append(dqj.copy())

    return single_obs, single_obs_dim

def main():

    # Load configuration
    config = load_config("h1_2.yaml")

    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']

    # Check number of joints
    n_joints = d.qpos.shape[0] - 7
    #print(f"Model DOFs (qpos): {d.qpos.shape[0]}, joints: {n_joints}, ctrl size: {d.ctrl.shape[0]}")
   # print(f"Robot has {n_joints} joints in MuJoCo model")

    # Initialize variables
    action = np.zeros(config['num_actions'], dtype=np.float32)
    target_dof_legs_pos = config['default_angles_legs'].copy()

    # cmd = config['cmd_init'].copy()
    # height_cmd = config['height_cmd']
    cmd = {
        "x": 0.0,
        "y": 0.0,
        "yaw": 0.0,
        "height": config["height_cmd"]
    }


    # print(config["legs_motor_pos_lower_limit_list"])
    # exit()
   # Initial observation
    fake_cmd_array = np.array([cmd["x"], cmd["y"], cmd["yaw"]])
    height_cmd = cmd["height"]


    # Initialize observation history as all zeros
    single_obs, single_obs_dim = compute_observation(d, config, action, fake_cmd_array, height_cmd, n_joints)


    obs_history = collections.deque(maxlen=config['obs_history_len'])
    for _ in range(config['obs_history_len']):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))

    # Prepare full observation vector
    obs = np.zeros(config['num_obs'], dtype=np.float32)

    # Load policy
    policy = torch.jit.load(config['policy_path'])
   # print(policy)
    counter = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < config['simulation_duration']:
            step_start = time.time()

            # --- Handle keyboard input ---
            cmd = handle_input(cmd)
            fake_cmd_array = np.array([cmd["x"], cmd["y"], cmd["yaw"]])
            height_cmd = cmd["height"]

            # Control leg joints with policy
            leg_tau = pd_control(
                target_dof_legs_pos,
                d.qpos[7:7+config['num_actions']],
                config['kps_legs'],

                np.zeros_like(config['kps_legs']),
                d.qvel[6:6+config['num_actions']],
                config['kds_legs']
            )

            # Assign torques (ensure correct shape)
            d.ctrl[:config['num_actions']] = leg_tau

            # Keep other joints at zero positions if they exist
            if n_joints > config['num_actions']:

                target_dof_arms_pos = config['default_angles_arms'].copy()

                arm_tau = pd_control(
                    target_dof_arms_pos,
                    d.qpos[7+config['num_actions']:7+n_joints],
                    config['kps_arms'],

                    np.zeros(n_joints-config['num_actions']),
                    d.qvel[6+config['num_actions']:6+n_joints],
                    config['kds_arms']
                )

                # TO DO - replace this with the actual max torque values from the model
                # Pass through a function
                # arm_tau = np.nan_to_num(arm_tau, nan=0.0, posinf=0.0, neginf=0.0)
                # arm_tau = np.clip(arm_tau, -max_tau, max_tau)
                d.ctrl[config['num_actions']:] = arm_tau

            # Step physics
            mujoco.mj_step(m, d)

            counter += 1
            if counter % config['control_decimation'] == 0:
                # Update observation
                single_obs, _ = compute_observation(d, config, action, cmd, height_cmd, n_joints)
                obs_history.append(single_obs)

                # Construct full observation with history
                for i, hist_obs in enumerate(obs_history):
                    start_idx = i * single_obs_dim
                    end_idx = start_idx + single_obs_dim
                    obs[start_idx:end_idx] = hist_obs

                # Policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                scaled_action = action * config['action_scale'] 
                clipped_action = np.clip(
                    scaled_action,
                    np.array(config["legs_motor_pos_lower_limit_list"]),
                    np.array(config["legs_motor_pos_upper_limit_list"]))
                
                # Transform action to target_dof_legs_pos
                target_dof_legs_pos =  clipped_action + config['default_angles_legs']
               
               
                target_dof_hist.append(target_dof_legs_pos.copy())
                t_hist.append(counter * config["simulation_dt"])

            # Sync viewer
            viewer.sync()

            # Time keeping
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    pygame.init()
    pygame.display.set_mode((300, 100))

    main()  # <-- Run simulation first, fills qpos_hist etc.

    # Convert to arrays AFTER simulation
    qpos_hist_arr = np.array(qpos_hist[::10])
    dqpos_hist_arr = np.array(dqpos_hist[::10])
    target_dof_hist_arr = np.array(target_dof_hist)
    t_hist_arr = np.array(t_hist)


    # Save raw data as pickle
    os.makedirs("logs", exist_ok=True)
    data = {
        "t": t_hist_arr,
        "qpos": qpos_hist_arr,
        "dqpos": dqpos_hist_arr,
        "actions": target_dof_hist_arr
    }
    with open("logs/joint_data.pkl", "wb") as f:
        pickle.dump(data, f)
    print("✅ Saved raw data to logs/joint_data.pkl")

    # Plots
    plot_qpos_vs_action(t_hist_arr, qpos_hist_arr, target_dof_hist_arr, joint_names, save_path="logs/qpos_vs_action.png")
    plot_dqpos(t_hist_arr, dqpos_hist_arr, joint_names, save_path="logs/dqpos.png")
