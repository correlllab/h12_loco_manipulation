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
import matplotlib.pyplot as plt

######################################################################
# Track key states manually
key_states = {
    "w": False, "s": False, "a": False, "d": False,
    "q": False, "e": False, "r": False, "f": False, "x": False,
}

# Histories for logging
qpos_hist, dqpos_hist, target_dof_hist, t_hist = [], [], [], []

# Joint names (12 dof, adapt if needed)
joint_names = [
    "L_hip_yaw", "L_hip_pitch", "L_hip_roll", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_yaw", "R_hip_pitch", "R_hip_roll", "R_knee", "R_ankle_pitch", "R_ankle_roll"
]

######################################################################
# Plotting functions

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

######################################################################
# Input handling

def handle_input(cmd, delta=0.005):
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

    # if key_states["w"]:
    #     cmd["x"] = min(cmd["x"] + delta, 1.5)
    # if key_states["s"]:
    #     cmd["x"] = max(cmd["x"] - delta, -1.5)
    # if key_states["d"]:
    #     cmd["y"] = min(cmd["y"] + delta, 1.0)
    # if key_states["a"]:
    #     cmd["y"] = max(cmd["y"] - delta, -1.0)
    # if key_states["q"]:
    #     cmd["yaw"] = min(cmd["yaw"] + delta, 0.5)
    # if key_states["e"]:
    #     cmd["yaw"] = max(cmd["yaw"] - delta, -0.5)

    if key_states["r"]:
        cmd["height"] = min(cmd["height"] + delta, 1.0)
    if key_states["f"]:
        cmd["height"] = max(cmd["height"] - delta, 0.65)
    if key_states["x"]:
        cmd = {"x":0.0, "y":0.0, "yaw":0.0, "height":1.0}
    return cmd

######################################################################
# Utilities

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    array_keys = ['kps_legs', 'kds_legs', 'default_angles_legs','kps_arms', 'kds_arms', 'default_angles_arms',
                  'cmd_scale', 'cmd_init', 'legs_motor_pos_lower_limit_list', 'legs_motor_pos_upper_limit_list']
    for key in array_keys:
        config[key] = np.array(config[key], dtype=np.float32)
    return config

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_rotate_inverse(q, v):
    w, x, y, z = q
    q_conj = np.array([w, -x, -y, -z])
    return np.array([
        v[0]*(q_conj[0]**2 + q_conj[1]**2 - q_conj[2]**2 - q_conj[3]**2) + v[1]*2*(q_conj[1]*q_conj[2] - q_conj[0]*q_conj[3]) + v[2]*2*(q_conj[1]*q_conj[3] + q_conj[0]*q_conj[2]),
        v[0]*2*(q_conj[1]*q_conj[2] + q_conj[0]*q_conj[3]) + v[1]*(q_conj[0]**2 - q_conj[1]**2 + q_conj[2]**2 - q_conj[3]**2) + v[2]*2*(q_conj[2]*q_conj[3] - q_conj[0]*q_conj[1]),
        v[0]*2*(q_conj[1]*q_conj[3] - q_conj[0]*q_conj[2]) + v[1]*2*(q_conj[2]*q_conj[3] + q_conj[0]*q_conj[1]) + v[2]*(q_conj[0]**2 - q_conj[1]**2 - q_conj[2]**2 + q_conj[3]**2)
    ])

def get_gravity_orientation(quat):
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)

def compute_observation(d, config, action, cmd, height_cmd, n_joints):
    qj = d.qpos[7:7+n_joints].copy()
    dqj = d.qvel[6:6+n_joints].copy()
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()
    default_joints = np.concatenate((config['default_angles_legs'], config['default_angles_arms']))
    qj_scaled = (qj - default_joints) * config['dof_pos_scale']
    dqj_scaled = dqj * config['dof_vel_scale']
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega * config['ang_vel_scale']
    cmd_array = np.array([cmd["x"], cmd["y"], cmd["yaw"]]) if isinstance(cmd, dict) else np.array(cmd)
    single_obs_dim = 3 + 1 + 3 + 3 + n_joints + n_joints + 12
    single_obs = np.zeros(single_obs_dim, dtype=np.float32)
    single_obs[0:3] = cmd_array* config['cmd_scale']
    single_obs[3:4] = np.array([height_cmd])
    single_obs[4:7] = omega_scaled
    single_obs[7:10] = gravity_orientation
    single_obs[10:10+n_joints] = qj_scaled
    single_obs[10+n_joints:10+2*n_joints] = dqj_scaled
    single_obs[10+2*n_joints:10+2*n_joints+12] = action
    return single_obs, single_obs_dim, qj.copy(), dqj.copy()

######################################################################

######################################################################
# Main simulation

def main():
    config = load_config("h1_2.yaml")
    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']
    n_joints = d.qpos.shape[0] - 7
    action = np.zeros(config['num_actions'], dtype=np.float32)
    # Start at default position, as the startup function will handle the transition
    target_dof_legs_pos = config['default_angles_legs'].copy()
    cmd = {"x":0.0, "y":0.0, "yaw":0.0, "height":config["height_cmd"]}
    height_cmd = cmd["height"]
    single_obs, single_obs_dim, _, _ = compute_observation(d, config, action, cmd, height_cmd, n_joints)
    obs_history = collections.deque(maxlen=config['obs_history_len'])
    for _ in range(config['obs_history_len']):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))
    obs = np.zeros(config['num_obs'], dtype=np.float32)
    policy = torch.jit.load(config['policy_path'])
    counter = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < config['simulation_duration']:
            step_start = time.time()
            cmd = handle_input(cmd)
            height_cmd = cmd["height"]

            # Apply PD control for legs based on policy action
            leg_tau = pd_control(
                target_dof_legs_pos,
                d.qpos[7:7+config['num_actions']],
                config['kps_legs'] * 1.33,
                np.zeros_like(config['kps_legs']),
                d.qvel[6:6+config['num_actions']],
                config['kds_legs']
            )
            d.ctrl[:config['num_actions']] = leg_tau
            
            # Apply PD control for arms to hold default position
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
                d.ctrl[config['num_actions']:] = arm_tau

            mujoco.mj_step(m, d)
            counter += 1

            # Get policy action at the specified control frequency
            if counter % config['control_decimation'] == 0:
                single_obs, _, qj, dqj = compute_observation(d, config, action, cmd, height_cmd, n_joints)
                obs_history.append(single_obs)
                obs = np.concatenate(list(obs_history))
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                scaled_action = action * config['action_scale']
                
                # Update target based on policy action
                target_dof_legs_pos = scaled_action + config['default_angles_legs']

                # Append histories for logging
                qpos_hist.append(qj)
                dqpos_hist.append(dqj)
                # Log the actual command sent to the PD controller
                target_dof_hist.append(target_dof_legs_pos.copy())
                t_hist.append(counter * config["simulation_dt"])

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

######################################################################
# Entry point
if __name__ == "__main__":
    pygame.init()
    pygame.display.set_mode((300, 100)) # Small window for pygame input
    main()

    # Ensure logs directory exists
    os.makedirs("logs/mujoco", exist_ok=True)
    if not t_hist:
        print("No data logged, skipping plots.")
        sys.exit()
    
    qpos_hist_arr = np.array(qpos_hist)
    dqpos_hist_arr = np.array(dqpos_hist)
    target_dof_hist_arr = np.array(target_dof_hist)
    t_hist_arr = np.array(t_hist)

    # Save raw data
    data = {"t": t_hist_arr, "qpos": qpos_hist_arr, "dqpos": dqpos_hist_arr, "actions": target_dof_hist_arr}
    with open("logs/mujoco/joint_data.pkl", "wb") as f:
        pickle.dump(data, f)
    print("✅ Saved raw data to logs/mujoco/joint_data.pkl")

    # Plots
    plot_qpos_vs_action(t_hist_arr, qpos_hist_arr, target_dof_hist_arr, joint_names, save_path="logs/mujoco/qpos_vs_action.png")
    plot_dqpos(t_hist_arr, dqpos_hist_arr, joint_names, save_path="logs/mujoco/dqpos.png")
