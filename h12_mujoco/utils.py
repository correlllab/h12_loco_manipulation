import pygame
import numpy as np
import yaml
import collections


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
    """Handle keyboard input for robot control commands."""
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
        cmd["height"] = min(cmd["height"] + delta, 1.03)
    if key_states["f"]:
        cmd["height"] = max(cmd["height"] - delta, 0.65)
    if key_states["x"]:
        cmd = {"x":0.0, "y":0.0, "yaw":0.0, "height":1.03}
    return cmd


######################################################################
# Configuration utilities
def load_config(config_path):
    """Load configuration from YAML file and convert arrays to numpy arrays."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    array_keys = ['kps_legs', 'kds_legs', 'default_angles_legs','kps_arms', 'kds_arms', 'default_angles_arms',
                  'cmd_scale', 'cmd_init', 'legs_motor_pos_lower_limit_list', 'legs_motor_pos_upper_limit_list']
    for key in array_keys:
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)
    return config


######################################################################
# Control utilities
def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Compute PD control torques."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_rotate_inverse(q, v):
    """Rotate vector v by inverse of quaternion q."""
    w, x, y, z = q
    q_conj = np.array([w, -x, -y, -z])
    # This calculation can be simplified or done with a library, but keeping as is from original
    return np.array([
        v[0]*(q_conj[0]**2 + q_conj[1]**2 - q_conj[2]**2 - q_conj[3]**2) + v[1]*2*(q_conj[1]*q_conj[2] - q_conj[0]*q_conj[3]) + v[2]*2*(q_conj[1]*q_conj[3] + q_conj[0]*q_conj[2]),
        v[0]*2*(q_conj[1]*q_conj[2] + q_conj[0]*q_conj[3]) + v[1]*(q_conj[0]**2 - q_conj[1]**2 + q_conj[2]**2 - q_conj[3]**2) + v[2]*2*(q_conj[2]*q_conj[3] - q_conj[0]*q_conj[1]),
        v[0]*2*(q_conj[1]*q_conj[3] - q_conj[0]*q_conj[2]) + v[1]*2*(q_conj[2]*q_conj[3] + q_conj[0]*q_conj[1]) + v[2]*(q_conj[0]**2 - q_conj[1]**2 - q_conj[2]**2 + q_conj[3]**2)
    ])


def get_gravity_orientation(quat):
    """Get gravity orientation in robot frame."""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)


