import time
import collections
import yaml
import torch
import numpy as np
import mujoco
import mujoco.viewer
import pygame

# Helper functions can remain outside the class as they are stateless
def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position and velocity commands."""
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_rotate_inverse(q, v):
    """Rotates vector v by the inverse of quaternion q."""
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    # Simplified calculation using intermediate products
    t = 2.0 * np.cross(q_conj[1:], v)
    v_prime = v + q_conj[0] * t + np.cross(q_conj[1:], t)
    return v_prime

def get_gravity_orientation(quat):
    """Calculates the gravity vector in the robot's base frame."""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)


class H1Controller:
    def __init__(self, config_path):
        """
        Initializes the controller, loads policies, and sets up the simulation.
        """
        # Load configuration from a single YAML file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # --- Load Model and Policies ---
        self.model = mujoco.MjModel.from_xml_path(self.config['xml_path'])
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.config['simulation_dt']
        
        self.squatting_policy = torch.jit.load(self.config['squatting_policy_path'])
        self.walking_policy = torch.jit.load(self.config['walking_policy_path'])
        
        # --- State Machine Initialization ---
        self.current_policy_mode = "SQUATTING" # Initial state
        print("✅ Controller initialized. Starting in SQUATTING mode.")

        # --- Initialize Robot State Variables ---
        self.num_leg_actions = self.config['num_leg_actions']
        self.num_total_actuators = self.model.nu
        self.default_angles_legs = np.array(self.config['default_angles_legs'])
        self.default_angles_arms = np.array(self.config['default_angles_arms'])
        
        self.action = np.zeros(self.num_leg_actions, dtype=np.float32)
        self.target_dof_pos_legs = self.default_angles_legs.copy()

        # --- PD Gains ---
        self.kps_legs = np.array(self.config['kps_legs'])
        self.kds_legs = np.array(self.config['kds_legs'])
        self.kps_arms = np.array(self.config['kps_arms'])
        self.kds_arms = np.array(self.config['kds_arms'])

        # --- Command and Observation Scaling ---
        self.dof_pos_scale = self.config['dof_pos_scale']
        self.dof_vel_scale = self.config['dof_vel_scale']
        self.ang_vel_scale = self.config['ang_vel_scale']
        self.action_scale = self.config['action_scale']
        self.walking_cmd_scale = np.array(self.config['walking_cmd_scale'])
        self.squatting_cmd_scale = np.array(self.config['squatting_cmd_scale'])

        # --- Command State ---
        self.cmd_vel = np.zeros(3, dtype=np.float32) # [x_vel, y_vel, yaw_vel]
        self.height_cmd = 1.0 # Start at standing height

        # --- Squatting Policy Observation History ---
        # The squatting policy requires a history of observations
        dummy_squat_obs = self._compute_squatting_obs(is_dummy=True)
        self.obs_history = collections.deque([dummy_squat_obs] * self.config['obs_history_len'], 
                                             maxlen=self.config['obs_history_len'])
        
        # --- Pygame for Keyboard Input ---
        pygame.init()
        pygame.display.set_mode((300, 100))
        pygame.display.set_caption("H1 Controller Input")
        self.key_states = {key: False for key in ['w', 'a', 's', 'd', 'r', 'f', 'x']}

    def handle_input(self):
        """
        Handles keyboard inputs to change commands and switch policies.
        """
        is_cmd_vel_triggered = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type in [pygame.KEYDOWN, pygame.KEYUP]:
                key_name = pygame.key.name(event.key)
                if key_name in self.key_states:
                    self.key_states[key_name] = (event.type == pygame.KEYDOWN)

        # Update commands based on key states
        # Walking commands
        self.cmd_vel[0] = 1.0 if self.key_states['w'] else (-1.0 if self.key_states['s'] else 0.0) # Forward/backward
        self.cmd_vel[2] = 1.0 if self.key_states['a'] else (-1.0 if self.key_states['d'] else 0.0) # Yaw left/right
        
        # Squatting commands
        if self.key_states['r']: self.height_cmd = min(self.height_cmd + 0.005, 1.0)
        if self.key_states['f']: self.height_cmd = max(self.height_cmd - 0.005, 0.65)
        
        # --- State Switching Logic ---
        if np.any(self.cmd_vel != 0) and self.current_policy_mode == "SQUATTING":
            self.current_policy_mode = "WALKING"
            print("🏃 Switched to WALKING mode.")
        
        if self.key_states['x']: # 'x' key resets to squatting mode
            self.cmd_vel[:] = 0.0
            self.height_cmd = 1.0
            if self.current_policy_mode == "WALKING":
                self.current_policy_mode = "SQUATTING"
                print("🧍 Switched back to SQUATTING mode.")


    def _compute_squatting_obs(self, is_dummy=False):
        """Generates the observation vector for the squatting policy."""
        if is_dummy: # For initialization
            return np.zeros(self.config['squatting_obs_dim'], dtype=np.float32)

        qj_legs = self.data.qpos[7 : 7 + self.num_leg_actions]
        dqj_legs = self.data.qvel[6 : 6 + self.num_leg_actions]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]
        
        # Note: Squatting policy uses a fixed cmd_vel of [0,0,0]
        cmd_for_obs = np.zeros(3) * self.squatting_cmd_scale
        height_for_obs = np.array([self.height_cmd])
        omega_scaled = omega * self.ang_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        qj_scaled = (qj_legs - self.default_angles_legs) * self.dof_pos_scale
        dqj_scaled = dqj_legs * self.dof_vel_scale
        
        obs = np.concatenate([
            cmd_for_obs, height_for_obs, omega_scaled, gravity_orientation,
            qj_scaled, dqj_scaled, self.action
        ]).astype(np.float32)
        
        return obs

    def _compute_walking_obs(self, counter):
        """Generates the observation vector for the walking policy."""
        qj_legs = self.data.qpos[7 : 7 + self.num_leg_actions]
        dqj_legs = self.data.qvel[6 : 6 + self.num_leg_actions]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]
        
        # Phase calculation
        period = 0.8
        phase = (counter * self.model.opt.timestep) % period / period
        sin_cos_phase = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])

        # Scale observations
        omega_scaled = omega * self.ang_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        cmd_scaled = self.cmd_vel * self.walking_cmd_scale
        qj_scaled = (qj_legs - self.default_angles_legs) * self.dof_pos_scale
        dqj_scaled = dqj_legs * self.dof_vel_scale
        
        obs = np.concatenate([
            omega_scaled, gravity_orientation, cmd_scaled, qj_scaled, dqj_scaled,
            self.action, sin_cos_phase
        ]).astype(np.float32)

        return obs

    def step(self, counter):
        """
        Computes the next action based on the current policy mode and state.
        """
        if self.current_policy_mode == "SQUATTING":
            single_obs = self._compute_squatting_obs()
            self.obs_history.append(single_obs)
            obs_flat = np.concatenate(list(self.obs_history))
            obs_tensor = torch.from_numpy(obs_flat).unsqueeze(0)
            self.action = self.squatting_policy(obs_tensor).detach().numpy().squeeze()
            
        elif self.current_policy_mode == "WALKING":
            obs = self._compute_walking_obs(counter)
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            self.action = self.walking_policy(obs_tensor).detach().numpy().squeeze()

        # Update target joint positions from the action
        scaled_action = self.action * self.action_scale
        
        # Clipping is good practice, assuming limits are in the config
        lower_limits = np.array(self.config['legs_motor_pos_lower_limit_list'])
        upper_limits = np.array(self.config['legs_motor_pos_upper_limit_list'])
        clipped_action = np.clip(scaled_action, lower_limits, upper_limits)
        
        self.target_dof_pos_legs = clipped_action + self.default_angles_legs
        
    def compute_torques(self):
        """Computes PD torques for all actuators."""
        # Leg torques
        tau_legs = pd_control(
            self.target_dof_pos_legs, self.data.qpos[7 : 7 + self.num_leg_actions], self.kps_legs,
            np.zeros(self.num_leg_actions), self.data.qvel[6 : 6 + self.num_leg_actions], self.kds_legs
        )
        self.data.ctrl[:self.num_leg_actions] = tau_legs
        
        # Arm torques (holding default position)
        if self.num_total_actuators > self.num_leg_actions:
            tau_arms = pd_control(
                self.default_angles_arms, self.data.qpos[7 + self.num_leg_actions:], self.kps_arms,
                np.zeros(self.num_total_actuators - self.num_leg_actions), 
                self.data.qvel[6 + self.num_leg_actions:], self.kds_arms
            )
            self.data.ctrl[self.num_leg_actions:] = tau_arms


if __name__ == "__main__":
    # The config file now drives the entire setup
    controller = H1Controller("h1_2_combined.yaml")

    counter = 0
    
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        start_time = time.time()
        sim_duration = controller.config['simulation_duration']

        while viewer.is_running() and time.time() - start_time < sim_duration:
            step_start_time = time.time()

            # Handle user input to update commands and potentially switch policies
            controller.handle_input()
            
            # Decide if it's time to run the policy inference
            if counter % controller.config['control_decimation'] == 0:
                controller.step(counter)

            # Always compute and apply torques at the simulation frequency
            controller.compute_torques()
            
            # Step the physics simulation
            mujoco.mj_step(controller.model, controller.data)
            counter += 1
            
            viewer.sync()

            # Maintain simulation real-time factor
            time_until_next_step = controller.model.opt.timestep - (time.time() - step_start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("✅ Simulation finished.")