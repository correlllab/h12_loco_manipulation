import time
import collections
import torch
import numpy as np
import mujoco
import mujoco.viewer
import rerun as rr

from utils import (
    load_config, pd_control, get_gravity_orientation, joint_names
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
        walk_params = self.config.get('policies', {}).get('walk')
        
        # Create merged config
        self.config = {**shared_params, **walk_params}
        
        # Set default values for missing parameters
        self.config['obs_history_len'] = 1  # Walk policy doesn't use history
        
        self.model = mujoco.MjModel.from_xml_path(self.config['xml_path'])
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.config['simulation_dt']
        
        # Robot state
        self.n_joints = self.data.qpos.shape[0] - 7 #remove 7 for base pos(3) + base quat(4)
        self.action = np.zeros(self.config['num_actions'], dtype=np.float32)
        self.target_dof_legs_pos = self.config.get('default_angles_legs').copy()
        
        # Command state for walking
        self.cmd_vel = np.zeros(3, dtype=np.float32)  # [x_vel, y_vel, yaw_vel]
        self.height_cmd = self.config.get("height_cmd")
        
        # Simulation state
        self.counter = 0
        
        # Observation and policy
        self.single_obs = self._compute_observation()
        self.policy = torch.jit.load(self.config['policy_path'])
        
        # # Initialize Rerun
        # rr.init("humanoid_walk_simulation", spawn=True)
    
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
        omega_scaled = omega * self.config.get('ang_vel_scale')
        gravity_orientation = get_gravity_orientation(quat)
        cmd_scaled = self.cmd_vel * self.config.get('cmd_scale')
        qj_scaled = (qj - self.config.get('default_angles_legs')) * self.config.get('dof_pos_scale')
        dqj_scaled = dqj * self.config.get('dof_vel_scale')
        
        # Walking observation structure: [omega, gravity, cmd, qj, dqj, action, phase]
        obs = np.concatenate([
            omega_scaled, gravity_orientation, cmd_scaled, qj_scaled, dqj_scaled,
            self.action, sin_cos_phase
        ]).astype(np.float32)
              
        return obs
    
    def step(self):
        """Execute one simulation step."""
        step_start = time.time()
        
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
    
    def _update_policy(self):
        """Update policy and compute new target positions."""
        try:
            # Compute observation
            self.single_obs = self._compute_observation()

            obs_tensor = torch.from_numpy(self.single_obs).unsqueeze(0)
            
            self.action = self.policy(obs_tensor).detach().numpy().squeeze()

            scaled_action = self.action * self.config['action_scale']

            lower_limits = self.config.get('legs_motor_pos_lower_limit_list')
            upper_limits = self.config.get('legs_motor_pos_upper_limit_list')

            clipped_action = np.clip(scaled_action, lower_limits, upper_limits)

            self.target_dof_legs_pos = clipped_action + self.config['default_angles_legs']

            # Log to Rerun
            self._log_to_rerun()
            
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
            
            # Log all components of commanded velocity
            rr.log(f"joint_tracking/{name}/CmdVel_X", rr.Scalar(self.cmd_vel[0]))  # Forward velocity (x)
            rr.log(f"joint_tracking/{name}/CmdVel_Y", rr.Scalar(self.cmd_vel[1]))  # Lateral velocity (y)
            rr.log(f"joint_tracking/{name}/CmdVel_Yaw", rr.Scalar(self.cmd_vel[2]))  # Rotational velocity (yaw/z)
            
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