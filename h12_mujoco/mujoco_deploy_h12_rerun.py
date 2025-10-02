import sys
import time
import collections
import yaml
import torch
import numpy as np
import mujoco
import mujoco.viewer
import pygame
import os
import rerun as rr

######################################################################
# Track key states manually
key_states = {
    "w": False, "s": False, "a": False, "d": False,
    "q": False, "e": False, "r": False, "f": False, "x": False,
}

# Joint names (12 dof, adapt if needed)
joint_names = [
    "L_hip_yaw", "L_hip_pitch", "L_hip_roll", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_yaw", "R_hip_pitch", "R_hip_roll", "R_knee", "R_ankle_pitch", "R_ankle_roll"
]

######################################################################
# Input handling
def handle_input(cmd, delta=0.0005):
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
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)
    return config

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_rotate_inverse(q, v):
    w, x, y, z = q
    q_conj = np.array([w, -x, -y, -z])
    # This calculation can be simplified or done with a library, but keeping as is from original
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
    default_joints = np.concatenate((config.get('default_angles_legs', []), config.get('default_angles_arms', [])))
    qj_scaled = (qj - default_joints) * config.get('dof_pos_scale', 1.0)
    dqj_scaled = dqj * config.get('dof_vel_scale', 1.0)
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega * config.get('ang_vel_scale', 1.0)
    cmd_array = np.array([cmd.get("x", 0.0), cmd.get("y", 0.0), cmd.get("yaw", 0.0)])
    single_obs_dim = 3 + 1 + 3 + 3 + n_joints + n_joints + 12
    single_obs = np.zeros(single_obs_dim, dtype=np.float32)
    single_obs[0:3] = cmd_array * config.get('cmd_scale', np.ones(3))
    single_obs[3:4] = np.array([height_cmd])
    single_obs[4:7] = omega_scaled
    single_obs[7:10] = gravity_orientation
    single_obs[10:10+n_joints] = qj_scaled
    single_obs[10+n_joints:10+2*n_joints] = dqj_scaled
    single_obs[10+2*n_joints:10+2*n_joints+12] = action
    return single_obs, single_obs_dim, qj.copy(), dqj.copy()
    
######################################################################
# Main simulation
def main():
    rr.init("humanoid_simulation", spawn=True)

    config = load_config("h1_2.yaml")
    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']
    n_joints = d.qpos.shape[0] - 7
    action = np.zeros(config['num_actions'], dtype=np.float32)
    target_dof_legs_pos = config.get('default_angles_legs', np.zeros(config['num_actions'])).copy()
    cmd = {"x":0.0, "y":0.0, "yaw":0.0, "height":config.get("height_cmd", 0.8)}
    height_cmd = cmd["height"]
    single_obs, _, _, _ = compute_observation(d, config, action, cmd, height_cmd, n_joints)
    obs_history = collections.deque(maxlen=config['obs_history_len'])
    for _ in range(config['obs_history_len']):
        obs_history.append(np.zeros_like(single_obs))
    policy = torch.jit.load(config['policy_path'])
    counter = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < config['simulation_duration']:
            step_start = time.time()
            cmd = handle_input(cmd)
            height_cmd = cmd["height"]

            leg_tau = pd_control(
                target_dof_legs_pos, d.qpos[7:7+config['num_actions']],
                config['kps_legs'], np.zeros_like(config['kps_legs']),
                d.qvel[6:6+config['num_actions']], config['kds_legs']
            )
            d.ctrl[:config['num_actions']] = leg_tau
            
            if n_joints > config['num_actions']:
                target_dof_arms_pos = config['default_angles_arms'].copy()
                arm_tau = pd_control(
                    target_dof_arms_pos, d.qpos[7+config['num_actions']:7+n_joints],
                    config['kps_arms'], np.zeros(n_joints-config['num_actions']),
                    d.qvel[6+config['num_actions']:6+n_joints], config['kds_arms']
                )
                d.ctrl[config['num_actions']:] = arm_tau

            mujoco.mj_step(m, d)
            counter += 1

            if counter % config['control_decimation'] == 0:
                single_obs, _, qj, dqj = compute_observation(d, config, action, cmd, height_cmd, n_joints)
                obs_history.append(single_obs)
                obs = np.concatenate(list(obs_history))
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                
                target_dof_legs_pos = (action * config['action_scale']) + config['default_angles_legs']

                current_time = counter * config["simulation_dt"]
                rr.set_time_seconds("sim_time", current_time)

                for i, name in enumerate(joint_names):
                    rr.log(f"joint_tracking/{name}/Actual", rr.Scalar(qj[i]))
                    rr.log(f"joint_tracking/{name}/Target", rr.Scalar(target_dof_legs_pos[i]))
                    rr.log(f"joint_tracking/{name}/HeightCmd", rr.Scalar(height_cmd))
                    rr.log(f"joint_velocity/{name}", rr.Scalar(dqj[i]))

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

######################################################################
# Entry point
if __name__ == "__main__":
    pygame.init()
    pygame.display.set_mode((300, 100))
    main()
    print("✅ Simulation finished. Check the Rerun viewer for plots.")