import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

   # parser = argparse.ArgumentParser()
   # parser.add_argument("config_file", type=str, help="config file name in the config folder")
   # args = parser.parse_args()
   # config_file = args.config_file
    with open(f"/home/niraj/isaac_projects/h12_loco_manipulation/h12_mujoco/h1_2_walk.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        # <<< MODIFIED: Load leg-specific PD gains and default angles
        # kps_legs = np.array(config["kps"], dtype=np.float32)
        # kds_legs = np.array(config["kds"], dtype=np.float32)
        # default_angles_legs = np.array(config["default_angles_legs"], dtype=np.float32)

        # <<< MODIFIED: Load upper body-specific PD gains and default angles
        kps_arms = np.array(config["kps_arms"], dtype=np.float32)
        kds_arms = np.array(config["kds_arms"], dtype=np.float32)
        default_angles_arms = np.array(config["default_angles_arms"], dtype=np.float32)

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        legs_motor_pos_lower_limit_list = config["legs_motor_pos_lower_limit_list"]
        legs_motor_pos_upper_limit_list = config["legs_motor_pos_upper_limit_list"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    num_total_actuators = m.nu


    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            tau_legs = pd_control(
                target_dof_pos, d.qpos[7 : 7 + num_actions],
                kps, np.zeros(num_actions),
                d.qvel[6 : 6 + num_actions], kds
            )
            d.ctrl[:num_actions] = tau_legs
            
            tau_upper_body = pd_control(
                default_angles_arms, d.qpos[7 + num_actions:],
                kps_arms, np.zeros(num_total_actuators - num_actions),
                d.qvel[6 + num_actions:], kds_arms
            )
            d.ctrl[num_actions:] = tau_upper_body
            
            #tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
           
            #d.ctrl[:] = tau
            
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:7+num_actions]
                dqj = d.qvel[6:6+num_actions]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                scaled_action = action * action_scale 

                clipped_action = np.clip(
                                    scaled_action,
                                    np.array(legs_motor_pos_lower_limit_list),
                                    np.array(legs_motor_pos_upper_limit_list))


                target_dof_pos = clipped_action + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
