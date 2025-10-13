import time
import collections
import torch
import numpy as np
import mujoco
import mujoco.viewer
import rerun as rr

from utils import (
    handle_input, load_config, pd_control, get_gravity_orientation, joint_names
)


class H12_Controller_Walk:
    """H12 Humanoid Robot Controller for MuJoCo simulation - Walking Policy."""
    
    def __init__(self, config_path="h1_2.yaml"):
        """Initialize the H12 walk controller.
        
        Args:
            config_path (str): Path to the configuration YAML file
        """
        self.config = load_config(config_path)
        
        # Merge shared params with walk policy params
        shared_params = self.config.get('shared_params', {})
        walk_params = self.config.get('policies', {}).get('walk', {})
        
        # Create merged config
        self.config = {**shared_params, **walk_params}
        
        # Set default values for missing parameters
        if 'obs_history_len' not in self.config:
            self.config['obs_history_len'] = 1  # Walk policy doesn't use history
        
        self.model = mujoco.MjModel.from_xml_path(self.config['xml_path'])
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.config['simulation_dt']
        
        # Robot state
        self.n_joints = self.data.qpos.shape[0] - 7
        self.action = np.zeros(self.config['num_actions'], dtype=np.float32)
        self.target_dof_legs_pos = self.config.get('default_angles_legs', np.zeros(self.config['num_actions'])).copy()
        
        # Command state for walking
        self.cmd_vel = np.zeros(3, dtype=np.float32)  # [x_vel, y_vel, yaw_vel]
        self.height_cmd = self.config.get("height_cmd", 0.8)
        
        # Simulation state
        self.counter = 0
        
        # Observation and policy
        self.single_obs = self._compute_observation()
        self.policy = torch.jit.load(self.config['policy_path'])
        
        # Initialize Rerun
        rr.init("humanoid_walk_simulation", spawn=True)
    
    def _compute_observation(self):
        """Compute observation vector for the walking policy."""
        qj = self.data.qpos[7:7+self.config['num_actions']].copy()
        dqj = self.data.qvel[6:6+self.config['num_actions']].copy()
        quat = self.data.qpos[3:7].copy()
        omega = self.data.qvel[3:6].copy()
        
        # Phase calculation for walking gait
        period = 0.8
        phase = (self.counter * self.model.opt.timestep) % period / period
        sin_cos_phase = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
        
        # Scale observations
        omega_scaled = omega * self.config.get('ang_vel_scale', 1.0)
        gravity_orientation = get_gravity_orientation(quat)
        cmd_scaled = self.cmd_vel * self.config.get('cmd_scale', np.ones(3))
        qj_scaled = (qj - self.config.get('default_angles_legs', np.zeros(self.config['num_actions']))) * self.config.get('dof_pos_scale', 1.0)
        dqj_scaled = dqj * self.config.get('dof_vel_scale', 1.0)
        
        # Walking observation structure: [omega, gravity, cmd, qj, dqj, action, phase]
        obs = np.concatenate([
            omega_scaled, gravity_orientation, cmd_scaled, qj_scaled, dqj_scaled,
            self.action, sin_cos_phase
        ]).astype(np.float32)
        
        # Debug print
        if self.counter % 100 == 0:  # Print every 100 steps
            print(f"🔍 Walk obs debug - counter: {self.counter}")
            print(f"🔍 obs shape: {obs.shape}")
            print(f"🔍 cmd_vel: {self.cmd_vel}")
            print(f"🔍 cmd_scaled: {cmd_scaled}")
            print(f"🔍 qj shape: {qj.shape}, dqj shape: {dqj.shape}")
            print(f"🔍 action shape: {self.action.shape}")
            print(f"🔍 sin_cos_phase: {sin_cos_phase}")
        
        return obs
    
    def step(self):
        """Execute one simulation step."""
        step_start = time.time()
        
        # Handle input (skip if pygame not available)
        try:
            self.cmd_vel = self._handle_walk_input()
        except ImportError:
            # If pygame is not available, keep current command
            pass
        
        # Compute leg torques using PD control
        leg_tau = pd_control(
            self.target_dof_legs_pos, 
            self.data.qpos[7:7+self.config['num_actions']],
            self.config['kps_legs'], 
            np.zeros_like(self.config['kps_legs']),
            self.data.qvel[6:6+self.config['num_actions']], 
            self.config['kds_legs']
        )
        self.data.ctrl[:self.config['num_actions']] = leg_tau
        
        # Compute arm torques if robot has arms
        if self.n_joints > self.config['num_actions']:
            target_dof_arms_pos = self.config['default_angles_arms'].copy()
            arm_tau = pd_control(
                target_dof_arms_pos, 
                self.data.qpos[7+self.config['num_actions']:7+self.n_joints],
                self.config['kps_arms'], 
                np.zeros(self.n_joints-self.config['num_actions']),
                self.data.qvel[6+self.config['num_actions']:6+self.n_joints], 
                self.config['kds_arms']
            )
            self.data.ctrl[self.config['num_actions']:] = arm_tau
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        self.counter += 1
        
        # Policy update at control decimation rate
        if self.counter % self.config['control_decimation'] == 0:
            self._update_policy()
        
        return step_start
    
    def _handle_walk_input(self):
        """Handle keyboard input for walking commands."""
        # This is a simplified version - you can expand this based on your needs
        cmd_vel = np.zeros(3, dtype=np.float32)
        
        # For now, return current command - you can add pygame input handling here
        # if needed, similar to the original walk_squat implementation
        return cmd_vel
    
    def _update_policy(self):
        """Update policy and compute new target positions."""
        try:
            # Compute observation
            print(f"🔍 Computing observation...")
            self.single_obs = self._compute_observation()
            print(f"🔍 Observation computed: shape={self.single_obs.shape}")
            
            # Get action from policy
            print(f"🔍 Converting to tensor...")
            obs_tensor = torch.from_numpy(self.single_obs).unsqueeze(0)
            print(f"🔍 Tensor shape: {obs_tensor.shape}")
            
            print(f"🔍 Running policy inference...")
            self.action = self.policy(obs_tensor).detach().numpy().squeeze()
            print(f"🔍 Policy output: shape={self.action.shape}, first 3 values={self.action[:3]}")
            
            # Update target positions with clipping
            print(f"🔍 Scaling action...")
            scaled_action = self.action * self.config['action_scale']
            print(f"🔍 Scaled action: first 3 values={scaled_action[:3]}")
            
            # Apply joint limits
            print(f"🔍 Applying joint limits...")
            lower_limits = self.config.get('legs_motor_pos_lower_limit_list', np.full(self.config['num_actions'], -np.pi))
            upper_limits = self.config.get('legs_motor_pos_upper_limit_list', np.full(self.config['num_actions'], np.pi))
            clipped_action = np.clip(scaled_action, lower_limits, upper_limits)
            print(f"🔍 Clipped action: first 3 values={clipped_action[:3]}")
            
            self.target_dof_legs_pos = clipped_action + self.config['default_angles_legs']
            print(f"🔍 Target positions: first 3 values={self.target_dof_legs_pos[:3]}")
            
            # Log to Rerun
            self._log_to_rerun()
            print(f"🔍 Policy update completed successfully")
            
        except Exception as e:
            print(f"❌ Error in _update_policy: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise
    
    def _log_to_rerun(self):
        """Log robot state to Rerun for visualization."""
        current_time = self.counter * self.config["simulation_dt"]
        rr.set_time_seconds("sim_time", current_time)
        
        qj = self.data.qpos[7:7+self.config['num_actions']].copy()
        dqj = self.data.qvel[6:6+self.config['num_actions']].copy()
        
        for i, name in enumerate(joint_names):
            rr.log(f"joint_tracking/{name}/Actual", rr.Scalar(qj[i]))
            rr.log(f"joint_tracking/{name}/Target", rr.Scalar(self.target_dof_legs_pos[i]))
            rr.log(f"joint_tracking/{name}/CmdVel", rr.Scalar(self.cmd_vel[0]))  # Forward velocity
            rr.log(f"joint_velocity/{name}", rr.Scalar(dqj[i]))
    
    def run_simulation(self):
        """Run the complete simulation."""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start = time.time()
            
            while viewer.is_running() and time.time() - start < self.config['simulation_duration']:
                step_start = self.step()
                
                # Sync viewer
                viewer.sync()
                
                # Maintain real-time simulation
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        print("✅ Walk simulation finished. Check the Rerun viewer for plots.")
    
    def reset(self):
        """Reset the controller to initial state."""
        self.data = mujoco.MjData(self.model)
        self.action = np.zeros(self.config['num_actions'], dtype=np.float32)
        self.target_dof_legs_pos = self.config.get('default_angles_legs', np.zeros(self.config['num_actions'])).copy()
        self.cmd_vel = np.zeros(3, dtype=np.float32)
        self.height_cmd = self.config.get("height_cmd", 0.8)
        self.counter = 0
        
        # Reset observation
        self.single_obs = self._compute_observation()
    
    def get_state(self):
        """Get current robot state.
        
        Returns:
            dict: Dictionary containing current robot state information
        """
        return {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'action': self.action.copy(),
            'target_dof_legs_pos': self.target_dof_legs_pos.copy(),
            'cmd_vel': self.cmd_vel.copy(),
            'height_cmd': self.height_cmd,
            'counter': self.counter
        }
    
    def set_command(self, x=0.0, y=0.0, yaw=0.0):
        """Set robot walking command.
        
        Args:
            x (float): Forward velocity command
            y (float): Lateral velocity command  
            yaw (float): Angular velocity command
        """
        self.cmd_vel[0] = x
        self.cmd_vel[1] = y
        self.cmd_vel[2] = yaw
