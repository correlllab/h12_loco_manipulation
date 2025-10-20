"""
Combined deploy for H12 real robot using split-policy controller wrappers.
This script reuses the RobotStateManager from `deploy_real.py` and uses the
new `h12_controller_squat` and `h12_controller_walk` wrappers so the file
structure matches the MuJoCo repository.
"""
import time
import argparse
import numpy as np
from common.remote_controller import RemoteController, KeyMap

# Import RobotStateManager from deploy_real (keeps hardware initialization centralized)
from deploy_real import Controller as RobotController

from h12_controller_squat import H12_Controller_Squat
from h12_controller_walk import H12_Controller_Walk


def main(net_if, config_path):
    # Initialize lower-level RobotController for DDS comms
    robot_ctrl = RobotController
    # For ease, reuse the existing Controller class: it expects a Config object;
    # to avoid duplicating parsing we will instruct users to run via existing deploy_real
    print("NOTE: This script provides a wrapper structure. For full hardware startup use deploy_real.py directly.")
    print("Creating policy wrappers...")

    squat = H12_Controller_Squat(config_path)
    walk = H12_Controller_Walk(config_path)

    # Basic loop example: run inference periodically and print target positions
    try:
        while True:
            # Run squat and walk inference as example (real code should select based on ModeManager)
            squat.infer_policy()
            walk.infer_policy()

            squat_targets = squat.get_target_positions()
            walk_targets = walk.get_target_positions()

            print(f"Squat target head: {squat_targets[:3]} | Walk target head: {walk_targets[:3]}")

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Shutdown requested, exiting.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net', type=str, help='network interface (if using ChannelFactoryInitialize)')
    parser.add_argument('--config', type=str, default='h1_2.yaml', help='path to config file')
    args = parser.parse_args()

    main(args.net, args.config)
