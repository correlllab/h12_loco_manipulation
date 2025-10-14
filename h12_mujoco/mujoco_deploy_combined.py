import pygame
import time
import numpy as np
import mujoco
import mujoco.viewer
import rerun as rr

from h12_controller_squat import H12_Controller_Squat
from h12_controller_walk import H12_Controller_Walk

from utils import pd_control

######################################################################
# Track key states manually
key_states = {
    "w": False, "s": False, "a": False, "d": False,
    "q": False, "e": False, "r": False, "f": False, "x": False,
}
# Input handling
def handle_combined_input(squat_controller, walk_controller):
    """Handle keyboard input and switch between controllers based on commands."""
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

    # Check for movement commands (walking)
    cmd_vel = np.zeros(3, dtype=np.float32)
    cmd_vel[0] = 0.3 if key_states['w'] else (-0.3 if key_states['s'] else 0.0)  # Forward/backward
    cmd_vel[1] = 0.3 if key_states['a'] else (-0.3 if key_states['d'] else 0.0)  # Yaw left/right    
    cmd_vel[2] = 0.3 if key_states['q'] else (-0.3 if key_states['e'] else 0.0)  # angular left/right

    # Check for height commands (squatting)
    height_cmd = squat_controller.height_cmd
    if key_states['r']:
        height_cmd = min(height_cmd + 0.0005, 1.0)
    if key_states['f']:
        height_cmd = max(height_cmd - 0.0005, 0.65)
    
    # Reset command
    if key_states['x']:
        cmd_vel[:] = 0.0
        height_cmd = 1.0
    
    # Update controllers
    walk_controller.set_command(cmd_vel[0], cmd_vel[1], cmd_vel[2])
    squat_controller.cmd["height"] = height_cmd
    squat_controller.height_cmd = height_cmd
    
    return cmd_vel, height_cmd

######################################################################
# Main simulation
def main():
    """Main function to run the H12 combined simulation."""
    # Initialize pygame for input handling
    pygame.init()
    pygame.display.set_mode((300, 100))
    pygame.display.set_caption("H12 Controller - WASDQE: Walk, RF: Squat, X: Reset")
    
    # Create both controllers
    squat_controller = H12_Controller_Squat("h1_2.yaml")
    walk_controller = H12_Controller_Walk("h1_2.yaml")
    
    print("✅ Controllers initialized.")
    print("🎮 Controls:")
    print("   WASDQE: Walking commands (forward/back, left/right, turn)")
    print("   R/F: Squatting commands (height control)")
    print("   X: Reset to squatting mode")
    print("🧍 Starting in SQUATTING mode...")
    
    # Start with squat controller (default state)
    current_controller = squat_controller
    current_mode = "SQUATTING"
    
    # Initialize Rerun for combined simulation
    rr.init("humanoid_combined_simulation", spawn=True)
    
    # Use squat controller's model and data for the main loop
    model = squat_controller.model
    data = squat_controller.data
    
    counter = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        simulation_duration = squat_controller.config['simulation_duration']
        
        while viewer.is_running() and time.time() - start_time < simulation_duration:
            step_start = time.time()
            
            # Handle input
            cmd_vel, height_cmd = handle_combined_input(squat_controller, walk_controller)
            
            # Simple logic:
            # 1. Start with squatting mode
            # 2. If WASDQE pressed -> switch to walking mode
            # 3. If R/F pressed -> stay in squatting mode for height control
            # 4. If X pressed -> reset to squatting mode
            
            cmd_vel_active = np.any(cmd_vel != 0)
            squat_input_active = key_states['r'] or key_states['f']
            
            # Determine mode based on input
            if cmd_vel_active:
                # WASDQE pressed -> switch to walking mode
                if current_mode != "WALKING":
                    current_mode = "WALKING"
                    # Force height to 1.0 for walking
                    squat_controller.height_cmd = 1.0
                    squat_controller.cmd["height"] = 1.0
                    print("🏃 WASDQE pressed -> WALKING mode (height forced to 1.0)")
            elif squat_input_active:
                # R/F pressed -> stay in squatting mode for height control
                if current_mode != "SQUATTING":
                    current_mode = "SQUATTING"
                    print("🧍 R/F pressed -> SQUATTING mode")
            elif key_states['x']:
                # X pressed -> reset to squatting mode
                current_mode = "SQUATTING"
                squat_controller.height_cmd = 1.0
                squat_controller.cmd["height"] = 1.0
                print("🧍 X pressed -> Reset to SQUATTING mode")
            
            # Single simulation step with policy switching
            step_start = time.time()
            
            # Handle input for current mode
            if current_mode == "WALKING":
                # Use walk controller's input handling
                walk_controller.cmd_vel = cmd_vel
                print(f"🔍 Setting walk_controller.cmd_vel = {cmd_vel}")
            else:
                # Use squat controller's input handling  
                squat_controller.cmd["height"] = height_cmd
                squat_controller.height_cmd = height_cmd
            
            # Compute torques based on current mode
            if current_mode == "WALKING":
                # Sync walk controller data with main simulation
                walk_controller.data = data
                walk_controller.counter = counter
                
                # Walking policy inference
                if counter % walk_controller.config['control_decimation'] == 0:
                    try:
                        # Debug observation before policy update
                        obs = walk_controller._compute_observation()
                        print(f"🔍 Walk obs shape: {obs.shape}, obs[0:5]: {obs[0:5]}")
                        print(f"🔍 Walk cmd_vel: {walk_controller.cmd_vel}")
                        print(f"🔍 Walk counter: {walk_controller.counter}")
                        
                        walk_controller._update_policy()
                        print(f"✅ Walking policy updated successfully at step {counter}")
                        print(f"🔍 Walk action: {walk_controller.action[:3]}...")
                    except Exception as e:
                        print(f"❌ Walking policy failed at step {counter}: {e}")
                        print(f"🔍 Error type: {type(e).__name__}")
                        import traceback
                        traceback.print_exc()
                
                # Apply walk controller torques
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
                if walk_controller.n_joints > walk_controller.config['num_actions']:
                    target_dof_arms_pos = walk_controller.config['default_angles_arms'].copy()
                    arm_tau = pd_control(
                        target_dof_arms_pos, 
                        data.qpos[7+walk_controller.config['num_actions']:7+walk_controller.n_joints],
                        walk_controller.config['kps_arms'], 
                        np.zeros(walk_controller.n_joints-walk_controller.config['num_actions']),
                        data.qvel[6+walk_controller.config['num_actions']:6+walk_controller.n_joints], 
                        walk_controller.config['kds_arms']
                    )
                    data.ctrl[walk_controller.config['num_actions']:] = arm_tau
            else:
                # Sync squat controller data with main simulation
                squat_controller.data = data
                squat_controller.counter = counter
                
                # Squatting policy inference
                if counter % squat_controller.config['control_decimation'] == 0:
                    squat_controller._update_policy()
                
                # Apply squat controller torques
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
                if squat_controller.n_joints > squat_controller.config['num_actions']:
                    target_dof_arms_pos = squat_controller.config['default_angles_arms'].copy()
                    arm_tau = pd_control(
                        target_dof_arms_pos, 
                        data.qpos[7+squat_controller.config['num_actions']:7+squat_controller.n_joints],
                        squat_controller.config['kps_arms'], 
                        np.zeros(squat_controller.n_joints-squat_controller.config['num_actions']),
                        data.qvel[6+squat_controller.config['num_actions']:6+squat_controller.n_joints], 
                        squat_controller.config['kds_arms']
                    )
                    data.ctrl[squat_controller.config['num_actions']:] = arm_tau
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            counter += 1
            
            # Log to Rerun
            current_time = counter * squat_controller.config["simulation_dt"]
            rr.set_time_seconds("sim_time", current_time)
            
            # Log mode information
            rr.log("mode/current_mode", rr.Scalar(1.0 if current_mode == "WALKING" else 0.0))
            rr.log("mode/cmd_vel_active", rr.Scalar(1.0 if cmd_vel_active else 0.0))
            
            # Sync viewer
            viewer.sync()
            
            # Maintain real-time simulation
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    print("✅ Combined simulation finished. Check the Rerun viewer for plots.")

######################################################################
# Entry point
if __name__ == "__main__":
    main()
