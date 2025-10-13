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


class H12_Controller_Squat:
    """H12 Humanoid Robot Controller for MuJoCo simulation."""
    
    def __init__(self, config_path="h1_2.yaml", policy_name="squat"):
        """Initialize the H12 controller.
        
        Args:
            config_path (str): Path to the configuration YAML file
            policy_name (str): Name of the policy to use ('squat' or 'walk')
        """
        self.config = load_config(config_path)
        
        # Merge shared params with policy-specific params
        shared_params = self.config.get('shared_params', {})
        policy_params = self.config.get('policies', {}).get(policy_name, {})
        
        # Create merged config
        self.config = {**shared_params, **policy_params}
        
        # Set default values for missing parameters
        if 'obs_history_len' not in self.config:
            self.config['obs_history_len'] = 1  # Default for walk policy
        self.model = mujoco.MjModel.from_xml_path(self.config['xml_path'])
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.config['simulation_dt']
        
        # Robot state
        self.n_joints = self.data.qpos.shape[0] - 7
        self.action = np.zeros(self.config['num_actions'], dtype=np.float32)
        self.target_dof_legs_pos = self.config.get('default_angles_legs', np.zeros(self.config['num_actions'])).copy()
        
        # Command state
        self.cmd = {
            "x": 0.0, 
            "y": 0.0, 
            "yaw": 0.0, 
            "height": self.config.get("height_cmd", 0.8)
        }
        self.height_cmd = self.cmd["height"]
        
        # Observation and policy
        self.single_obs, _, _, _ = self._compute_observation()
        self.obs_history = self._create_obs_history()
        self.policy = torch.jit.load(self.config['policy_path'])
        
        # Simulation state
        self.counter = 0
        
        # Initialize Rerun
        rr.init("humanoid_simulation", spawn=True)
    
    def _compute_observation(self):
        """Compute observation vector for the policy."""
        qj = self.data.qpos[7:7+self.n_joints].copy()
        dqj = self.data.qvel[6:6+self.n_joints].copy()
        quat = self.data.qpos[3:7].copy()
        omega = self.data.qvel[3:6].copy()
        default_joints = np.concatenate((self.config.get('default_angles_legs', []), self.config.get('default_angles_arms', [])))
        qj_scaled = (qj - default_joints) * self.config.get('dof_pos_scale', 1.0)
        dqj_scaled = dqj * self.config.get('dof_vel_scale', 1.0)
        gravity_orientation = get_gravity_orientation(quat)
        omega_scaled = omega * self.config.get('ang_vel_scale', 1.0)
        cmd_array = np.array([self.cmd.get("x", 0.0), self.cmd.get("y", 0.0), self.cmd.get("yaw", 0.0)])
        single_obs_dim = 3 + 1 + 3 + 3 + self.n_joints + self.n_joints + 12
        single_obs = np.zeros(single_obs_dim, dtype=np.float32)
        single_obs[0:3] = cmd_array * self.config.get('cmd_scale', np.ones(3))
        single_obs[3:4] = np.array([self.height_cmd])
        single_obs[4:7] = omega_scaled
        single_obs[7:10] = gravity_orientation
        single_obs[10:10+self.n_joints] = qj_scaled
        single_obs[10+self.n_joints:10+2*self.n_joints] = dqj_scaled
        single_obs[10+2*self.n_joints:10+2*self.n_joints+12] = self.action
        return single_obs, single_obs_dim, qj.copy(), dqj.copy()
    
    def _create_obs_history(self):
        """Create observation history deque."""
        obs_history = collections.deque(maxlen=self.config['obs_history_len'])
        for _ in range(self.config['obs_history_len']):
            obs_history.append(np.zeros_like(self.single_obs))
        return obs_history
    
    def step(self):
        """Execute one simulation step."""
        step_start = time.time()
        
        # Handle input (skip if pygame not available)
        try:
            self.cmd = handle_input(self.cmd)
            self.height_cmd = self.cmd["height"]
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
    
    def _update_policy(self):
        """Update policy and compute new target positions."""
        # Compute observation
        self.single_obs, _, qj, dqj = self._compute_observation()
        
        # Update observation history
        self.obs_history.append(self.single_obs)
        obs = np.concatenate(list(self.obs_history))
        
        # Get action from policy
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        
        # Update target positions
        self.target_dof_legs_pos = (self.action * self.config['action_scale']) + self.config['default_angles_legs']
        
        # Log to Rerun
        self._log_to_rerun(qj, dqj)
    
    def _log_to_rerun(self, qj, dqj):
        """Log robot state to Rerun for visualization."""
        current_time = self.counter * self.config["simulation_dt"]
        rr.set_time_seconds("sim_time", current_time)
        
        for i, name in enumerate(joint_names):
            rr.log(f"joint_tracking/{name}/Actual", rr.Scalar(qj[i]))
            rr.log(f"joint_tracking/{name}/Target", rr.Scalar(self.target_dof_legs_pos[i]))
            rr.log(f"joint_tracking/{name}/HeightCmd", rr.Scalar(self.height_cmd))
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
        
        print("✅ Simulation finished. Check the Rerun viewer for plots.")
    
    def reset(self):
        """Reset the controller to initial state."""
        self.data = mujoco.MjData(self.model)
        self.action = np.zeros(self.config['num_actions'], dtype=np.float32)
        self.target_dof_legs_pos = self.config.get('default_angles_legs', np.zeros(self.config['num_actions'])).copy()
        self.cmd = {
            "x": 0.0, 
            "y": 0.0, 
            "yaw": 0.0, 
            "height": self.config.get("height_cmd", 0.8)
        }
        self.height_cmd = self.cmd["height"]
        self.counter = 0
        
        # Reset observation history
        self.single_obs, _, _, _ = self._compute_observation()
        self.obs_history = self._create_obs_history()
    
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
            'cmd': self.cmd.copy(),
            'height_cmd': self.height_cmd,
            'counter': self.counter
        }
    
    def set_command(self, x=0.0, y=0.0, yaw=0.0, height=None):
        """Set robot command.
        
        Args:
            x (float): Forward velocity command
            y (float): Lateral velocity command  
            yaw (float): Angular velocity command
            height (float): Height command (if None, keeps current height)
        """
        self.cmd["x"] = x
        self.cmd["y"] = y
        self.cmd["yaw"] = yaw
        if height is not None:
            self.cmd["height"] = height
            self.height_cmd = height
