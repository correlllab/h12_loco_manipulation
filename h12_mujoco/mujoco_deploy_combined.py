"""
H12 Humanoid Robot - Combined Squat & Walk Controller
======================================================
This module handles switching between squatting and walking modes with smooth transitions.

Controls:
  - WASD: Walking commands (W/S: forward/back, A/D: left/right yaw)
  - Q/E: Rotation commands
  - R/F: Height adjustment (squatting mode)
  - X: Reset to default state
"""

import pygame
import time
import numpy as np
import mujoco
import mujoco.viewer
import rerun as rr

from h12_controller_squat import H12_Controller_Squat
from h12_controller_walk import H12_Controller_Walk
from utils import pd_control, key_states


######################################################################
# CONSTANTS & CONFIGURATION
######################################################################

# Height transition settings
HEIGHT_TRANSITION_RATE = 0.0008  # Height change per simulation step
MIN_HEIGHT = 0.65  # Minimum crouching height
MAX_HEIGHT = 1.04  # Maximum standing height
DEFAULT_HEIGHT = 1.04  # Default height for walking

# Mode definitions
MODE_SQUAT = "SQUATTING"
MODE_WALK = "WALKING"
MODE_TRANSITION = "TRANSITIONING" 

######################################################################
# INPUT HANDLING
######################################################################

class InputHandler:
    """Handles keyboard input and mode switching logic."""
    
    def __init__(self):
        self.cmd_vel = np.zeros(3, dtype=np.float32)
        self.height_cmd = MAX_HEIGHT
        
    def update(self):
        """Process keyboard events."""
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
    
    def get_movement_command(self):
        """Get velocity commands from WASD keys."""
        cmd_vel = np.zeros(3, dtype=np.float32)
        cmd_vel[0] = 0.5 if key_states['w'] else (-0.5 if key_states['s'] else 0.0)  # Forward/back
        cmd_vel[1] = 0.3 if key_states['a'] else (-0.3 if key_states['d'] else 0.0)  # sideways
        cmd_vel[2] = 0.5 if key_states['q'] else (-0.5 if key_states['e'] else 0.0)  # YAW
        return cmd_vel
    
    def get_height_command(self):
        """Get height adjustment commands from R/F keys."""
        delta_height = 0.0
        if key_states['r']:
            delta_height = HEIGHT_TRANSITION_RATE 
        elif key_states['f']:
            delta_height = -HEIGHT_TRANSITION_RATE
        return delta_height
    
    def is_reset_pressed(self):
        """Check if reset button is pressed."""
        return key_states['x']
    
    def is_movement_active(self):
        """Check if any movement command is active."""
        movement = self.get_movement_command()
        return np.any(movement != 0)
    
    def is_height_adjustment_active(self):
        """Check if manual height adjustment is active."""
        return key_states['r'] or key_states['f']


######################################################################
# STATE MACHINE
######################################################################

class ModeManager:
    """Manages robot mode transitions and state."""
    
    def __init__(self):
        self.current_mode = MODE_SQUAT
        self.target_height = MAX_HEIGHT
        self.current_height = MAX_HEIGHT
        self.transition_steps = 0
        self.max_transition_steps = int(0.5 / 0.002)  # 0.5 seconds at 500Hz
        self.transition_target_mode = MODE_SQUAT  # Track which mode to transition to
    
    def update_mode(self, movement_active, height_adjust_active, reset_pressed):
        """Update mode based on user input."""
        if reset_pressed:
            self._transition_to_squat()
            return
        
        if movement_active:
            # Start walking - initiate smooth transition to standing height
            if self.current_mode != MODE_WALK:
                self._start_transition_to_walk()
        elif height_adjust_active:
            # Manual height control - stay in squat mode
            if self.current_mode != MODE_SQUAT:
                self._transition_to_squat()
    
    def _start_transition_to_walk(self):
        """Begin transition to walking mode."""
        self.current_mode = MODE_TRANSITION
        self.target_height = DEFAULT_HEIGHT
        self.transition_target_mode = MODE_WALK
        self.transition_steps = 0
    
    def _transition_to_squat(self):
        """Transition to squatting mode."""
        self.current_mode = MODE_SQUAT
        self.transition_target_mode = MODE_SQUAT
        self.target_height = self.current_height  # Keep current height
    
    def _start_transition_to_squat_init(self, init_height):
        """Begin smooth transition back to squat initial state."""
        self.current_mode = MODE_TRANSITION
        self.target_height = init_height
        self.transition_target_mode = MODE_SQUAT
        self.transition_steps = 0
    
    def update_height_smoothly(self):
        """Smoothly interpolate height during transitions."""
        if self.current_mode == MODE_TRANSITION:
            # Gradually interpolate height towards target
            if abs(self.current_height - self.target_height) > 0.001:
                direction = 1.0 if self.target_height > self.current_height else -1.0
                self.current_height += direction * HEIGHT_TRANSITION_RATE
                self.transition_steps += 1
            else:
                # Transition complete
                self.current_height = self.target_height
                # Complete transition to target mode (walk or squat)
                if self.transition_target_mode == MODE_WALK:
                    self.current_mode = MODE_WALK
                else:
                    self.current_mode = MODE_SQUAT
        
        # Clamp height to valid range
        self.current_height = np.clip(self.current_height, MIN_HEIGHT, MAX_HEIGHT)
    
    def apply_manual_height_adjustment(self, delta_height):
        """Apply manual height adjustments (R/F keys)."""
        if self.current_mode == MODE_SQUAT:
            self.current_height = np.clip(
                self.current_height + delta_height,
                MIN_HEIGHT, MAX_HEIGHT
            )


######################################################################
# TORQUE APPLICATION
######################################################################

class TorqueController:
    """Applies computed torques to the robot."""
    
    @staticmethod
    def apply_squat_torques(squat_controller, data):
        """Apply torques from squat controller."""
        # Leg torques
        leg_tau = pd_control(
            squat_controller.target_dof_legs_pos,
            data.qpos[7:7+squat_controller.config['num_actions']],
            squat_controller.config['kps_legs'],
            np.zeros_like(squat_controller.config['kps_legs']),
            data.qvel[6:6+squat_controller.config['num_actions']],
            squat_controller.config['kds_legs']
        )
        data.ctrl[:squat_controller.config['num_actions']] = leg_tau
        
        # Arm torques
        arm_tau = pd_control(
            squat_controller.config['default_angles_arms'].copy(),
            data.qpos[7+squat_controller.config['num_actions']:7+squat_controller.n_joints],
            squat_controller.config['kps_arms'],
            np.zeros(squat_controller.n_joints - squat_controller.config['num_actions']),
            data.qvel[6+squat_controller.config['num_actions']:6+squat_controller.n_joints],
            squat_controller.config['kds_arms']
        )
        data.ctrl[squat_controller.config['num_actions']:] = arm_tau
    
    @staticmethod
    def apply_walk_torques(walk_controller, data):
        """Apply torques from walk controller."""
        # Leg torques
        leg_tau = pd_control(
            walk_controller.target_dof_legs_pos,
            data.qpos[7:7+walk_controller.config['num_actions']],
            walk_controller.config['kps_legs'],
            np.zeros_like(walk_controller.config['kps_legs']),
            data.qvel[6:6+walk_controller.config['num_actions']],
            walk_controller.config['kds_legs']
        )
        data.ctrl[:walk_controller.config['num_actions']] = leg_tau
        
        # Arm torques
        arm_tau = pd_control(
            walk_controller.config['default_angles_arms'].copy(),
            data.qpos[7+walk_controller.config['num_actions']:7+walk_controller.n_joints],
            walk_controller.config['kps_arms'],
            np.zeros(walk_controller.n_joints - walk_controller.config['num_actions']),
            data.qvel[6+walk_controller.config['num_actions']:6+walk_controller.n_joints],
            walk_controller.config['kds_arms']
        )
        data.ctrl[walk_controller.config['num_actions']:] = arm_tau


######################################################################
# POLICY INFERENCE
######################################################################

class PolicyExecutor:
    """Handles policy inference and updates."""
    
    @staticmethod
    def update_squat_policy(squat_controller, counter):
        """Execute squat policy inference."""
        if counter % squat_controller.config['control_decimation'] == 0:
            try:
                squat_controller._update_policy()
            except Exception as e:
                print(f"❌ Squat policy failed at step {counter}: {e}")
                import traceback
                traceback.print_exc()
    
    @staticmethod
    def update_walk_policy(walk_controller, counter):
        """Execute walk policy inference."""
        if counter % walk_controller.config['control_decimation'] == 0:
            try:
                walk_controller._update_policy()
            except Exception as e:
                print(f"❌ Walk policy failed at step {counter}: {e}")
                import traceback
                traceback.print_exc()


######################################################################
# LOGGING & VISUALIZATION
######################################################################

def log_to_rerun(mode, cmd_vel_active, height, counter, config):
    """Log simulation state to Rerun for visualization."""
    current_time = counter * config["simulation_dt"]
    rr.set_time_seconds("sim_time", current_time)
    
    mode_value = 1.0 if mode == MODE_WALK else 0.0
    rr.log("mode/current_mode", rr.Scalar(mode_value))
    rr.log("mode/cmd_vel_active", rr.Scalar(1.0 if cmd_vel_active else 0.0))
    rr.log("state/height", rr.Scalar(height))


def print_control_info():
    """Print control instructions."""
    print("\n" + "="*60)
    print("🤖 H12 HUMANOID - COMBINED SQUAT & WALK CONTROLLER")
    print("="*60)
    print("🎮 CONTROLS:")
    print("   W/S: Move forward/backward")
    print("   A/D: Turn left/right")
    print("   Q/E: Roll left/right")
    print("   R/F: Height adjustment (squat mode)")
    print("   X: Reset to default state")
    print("="*60)
    print("📍 Starting in SQUATTING mode...")
    print("="*60 + "\n")


def print_mode_change(old_mode, new_mode, reason=""):
    """Print mode change notification."""
    if old_mode != new_mode:
        emoji = "🏃" if new_mode == MODE_WALK else "🧍"
        transition = "🔄" if new_mode == MODE_TRANSITION else ""
        print(f"{emoji}{transition} Mode: {old_mode} → {new_mode} {reason}")


######################################################################
# MAIN SIMULATION LOOP
######################################################################

def main():
    """Main function to run the H12 combined simulation."""
    # Initialize pygame for input handling
    pygame.init()
    pygame.display.set_mode((300, 100))
    pygame.display.set_caption("H12 Controller - WASDQE: Walk, RF: Squat, X: Reset")
    
    print_control_info()
    
    # Initialize components
    squat_controller = H12_Controller_Squat("h1_2.yaml")
    walk_controller = H12_Controller_Walk("h1_2.yaml")

    input_handler = InputHandler()
    mode_manager = ModeManager()
    torque_controller = TorqueController()
    policy_executor = PolicyExecutor()
    
    print("✅ Controllers initialized.")
    
    # Initialize Rerun for visualization
    rr.init("humanoid_combined_simulation", spawn=True)
    
    # Shared simulation state
    model = squat_controller.model
    data = squat_controller.data
    counter = 0
    
    # Track mode for notifications
    last_printed_mode = mode_manager.current_mode
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        simulation_duration = squat_controller.config['simulation_duration']
        
        while viewer.is_running() and time.time() - start_time < simulation_duration:
            step_start = time.time()
            
            # ============= INPUT PROCESSING =============
            input_handler.update()
            movement_active = input_handler.is_movement_active()
            height_adjust_active = input_handler.is_height_adjustment_active()
            reset_pressed = input_handler.is_reset_pressed()
            
            # ============= MODE MANAGEMENT =============
            if reset_pressed:
                # Reset: smoothly transition to initial squatting state
                init_height = squat_controller.config.get('height_cmd', DEFAULT_HEIGHT)
                mode_manager._start_transition_to_squat_init(init_height)
                
                # Stop walking and reset commands
                squat_controller.cmd = {"x": 0.0, "y": 0.0, "yaw": 0.0, "height": init_height}
                walk_controller.cmd_vel[:] = 0.0
            else:
                mode_manager.update_mode(movement_active, height_adjust_active, reset_pressed)

            # Smooth height interpolation
            mode_manager.update_height_smoothly()
            
            # Print mode changes
            if mode_manager.current_mode != last_printed_mode:
                print_mode_change(last_printed_mode, mode_manager.current_mode)
                last_printed_mode = mode_manager.current_mode
            
            # ============= COMMAND UPDATES =============
            # Update height command
            squat_controller.height_cmd = mode_manager.current_height
            squat_controller.cmd["height"] = mode_manager.current_height
            
            # Apply manual height adjustment in squat mode
            delta_height = input_handler.get_height_command()
            if delta_height != 0:
                mode_manager.apply_manual_height_adjustment(delta_height)
                squat_controller.height_cmd = mode_manager.current_height
                squat_controller.cmd["height"] = mode_manager.current_height
            
            # Update walk velocity command
            cmd_vel = input_handler.get_movement_command()
            walk_controller.cmd_vel = cmd_vel
            
            # ============= POLICY INFERENCE =============
            if mode_manager.current_mode == MODE_WALK:
                walk_controller.data = data
                walk_controller.counter = counter
                policy_executor.update_walk_policy(walk_controller, counter)
                torque_controller.apply_walk_torques(walk_controller, data)
            else:  # SQUAT or TRANSITION modes
                squat_controller.data = data
                squat_controller.counter = counter
                policy_executor.update_squat_policy(squat_controller, counter)
                torque_controller.apply_squat_torques(squat_controller, data)
            
            # ============= SIMULATION STEP =============
            mujoco.mj_step(model, data)
            counter += 1
            
            # ============= LOGGING & VISUALIZATION =============
            log_to_rerun(
                mode_manager.current_mode,
                movement_active,
                mode_manager.current_height,
                counter,
                squat_controller.config
            )
            
            # Sync viewer
            viewer.sync()
            
            # ============= TIMING =============
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    print("\n✅ Simulation finished. Check the Rerun viewer for plots.")


if __name__ == "__main__":
    main()
