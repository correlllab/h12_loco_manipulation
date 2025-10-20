# H12 Combined Controller - Code Summary

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│         MAIN SIMULATION LOOP (500 Hz)                       │
└─────────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ↓                 ↓                 ↓
    ┌────────┐         ┌────────┐       ┌────────┐
    │ INPUT  │         │ MODE   │       │COMMANDS│
    │HANDLER │         │MANAGER │       │ UPDATE │
    └────────┘         └────────┘       └────────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           ↓
                  ┌─────────────────┐
                  │POLICY EXECUTOR  │
                  │(Squat or Walk)  │
                  └─────────────────┘
                           │
                  ┌─────────────────┐
                  │TORQUE CONTROLLER│
                  │(PD Control)     │
                  └─────────────────┘
                           │
                  ┌─────────────────┐
                  │MUJOCO PHYSICS   │
                  └─────────────────┘
```

---

## Components

| Component | Purpose | Key Methods |
|-----------|---------|------------|
| **InputHandler** | Reads keyboard (WASDQE, RF, X) | `get_movement_command()`, `get_height_command()` |
| **ModeManager** | State machine (SQUAT/WALK/TRANSITION) | `update_mode()`, `update_height_smoothly()` |
| **TorqueController** | Applies PD control torques | `apply_squat_torques()`, `apply_walk_torques()` |
| **PolicyExecutor** | Runs neural networks | `update_squat_policy()`, `update_walk_policy()` |
| **H12_Controller_Squat** | Squat policy wrapper | `_compute_observation()`, `_update_policy()` |
| **H12_Controller_Walk** | Walk policy wrapper | `_compute_observation()`, `_update_policy()` |

---

## Mode State Machine

```
                    SQUAT
                  ↙     ↖
            R/F key    W key (movement)
              ↙           ↖
         (adjust)      (initiate walk)
            ↙               ↖
         SQUAT         TRANSITION
                           │ (smooth rise 0.5s)
                           ↓
                         WALK
                           │
                    (release W)
                           ↓
                      TRANSITION
                           │ (smooth fall)
                           ↓
                         SQUAT

RESET (X key): TRANSITION → SQUAT (smooth to init height)
```

---

## Control Mapping

| Input | Action |
|-------|--------|
| **W/S** | Forward/backward velocity |
| **A/D** | Left/right yaw (0.3 units) |
| **Q/E** | Roll commands |
| **R/F** | Height ±0.0008 (squat mode only) |
| **X** | Smooth reset to squat init state |

---

## Height Transitions

```
Walking Mode (WASD):
  Current → TRANSITION → Default Height (1.04m) → WALK
            (0.5 sec smooth rise)

Squat Mode (Release WASD):
  Current → TRANSITION → Current → SQUAT
            (0.5 sec smooth fall)

Reset (X):
  Any Height → TRANSITION → Init Height (from config) → SQUAT
               (0.5 sec smooth transition)
```

---

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `HEIGHT_TRANSITION_RATE` | 0.0008 | Height change per step |
| `MIN_HEIGHT` | 0.65m | Lowest crouch |
| `MAX_HEIGHT` | 1.04m | Standing height |
| `DEFAULT_HEIGHT` | 1.04m | Walk mode target |
| `max_transition_steps` | 250 | 0.5 seconds @ 500Hz |
| `simulation_dt` | 0.002s | MuJoCo timestep |
| `control_decimation` | 10 | Policy runs @ 50Hz |

---

## Data Flow per Step

```
1. Read Input (keyboard)
   ↓
2. Update Mode & Smooth Height
   ↓
3. Set Controller Commands
   ↓
4. Run Policy (every 10 steps)
   ↓
5. Compute Torques (PD control)
   ↓
6. Step Physics (MuJoCo)
   ↓
7. Log & Visualize (Rerun)
   ↓
8. Sleep until next step (2ms)
```

---

## Observation Structure

### Squat Policy
- Commands (3): x, y, yaw
- Height (1): height_cmd
- IMU (6): omega (3) + gravity (3)
- Joints (12): positions
- Joints (12): velocities
- Action (12): previous
- **History**: 6 frames concatenated → 456 dims total

### Walk Policy
- Omega (3)
- Gravity (3)
- Commands (3): x, y, yaw
- Joints (12): positions
- Joints (12): velocities
- Action (12): previous
- Phase (2): sin/cos of gait phase
- **Total**: 47 dims (single frame, no history)

---

## Safety & Stability

✅ **Smooth Transitions**: All height changes interpolated over 0.5s  
✅ **Proper State Tracking**: Mode transitions complete before switching policies  
✅ **Error Handling**: Policy failures logged but don't crash  
✅ **Reset Functionality**: Smoothly returns to configured init state  
✅ **PD Control**: Joint targets bounded to valid limits  

---

## Performance

- **Simulation Frequency**: 500 Hz (dt=0.002s per step)
- **Policy Update Rate**: 50 Hz (control_decimation=10 → policy runs every 10 sim steps)
- **Transition Time**: 0.5 seconds = 250 simulation steps
- **Memory**: Shared MuJoCo data between controllers
- **CPU**: ~1 core for simulation + policy inference

---

## Summary

**Purpose**: Switch between squatting and walking with smooth, stable transitions  
**Inputs**: Keyboard (WASDQE, RF, X)  
**Outputs**: Joint torques → MuJoCo physics  
**Key Innovation**: Gradual height interpolation prevents instability  
**Status**: Fully functional, production-ready ✅
