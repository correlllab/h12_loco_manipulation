from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class H12RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.0] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint': 0,
            'left_hip_roll_joint': 0,
            'left_hip_pitch_joint': -0.16,
            'left_knee_joint': 0.36,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,

            'right_hip_yaw_joint': 0,
            'right_hip_roll_joint': 0,
            'right_hip_pitch_joint': -0.16,
            'right_knee_joint': 0.36,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,

            'torso_joint': 0,

            'left_shoulder_pitch_joint': 0.4,
            'left_shoulder_roll_joint': 0,
            'left_shoulder_yaw_joint': 0,
            'left_elbow_pitch_joint': 0.3,
            'left_elbow_roll_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,

            'right_shoulder_pitch_joint': 0.4,
            'right_shoulder_roll_joint': 0,
            'right_shoulder_yaw_joint': 0,
            'right_elbow_pitch_joint': 0.3,
            'right_elbow_roll_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'M'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle_pitch': 60,
                     'ankle_roll': 40,

                     "torso": 600,

                     "shoulder_pitch": 80,
                     "shoulder_roll": 80,
                     "shoulder_yaw": 40,
                     "wrist": 60,
                     "elbow": 40,

                     }  # [N*m/rad]
        damping = {  'hip_yaw': 5.,
                     'hip_roll': 5.,
                     'hip_pitch': 5.,
                     'knee': 7.5,
                     'ankle_pitch': 1.,
                     'ankle_roll': 0.3,
                     "torso": 15,
                     "shoulder_pitch": 2.,
                     "shoulder_roll": 2.,
                     "shoulder_yaw": 1.,
                     "wrist": 0.5,
                     "elbow": 1,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0

    class commands( LeggedRobotCfg.commands ):
        curriculum = False # NOTE set True later
        max_curriculum = 1.4
        num_commands = 5 # lin_vel_x, lin_vel_y, ang_vel_yaw, heading, height, orientation
        resampling_time = 4. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        heading_to_ang_vel = False
        class ranges( LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0.0, 0.0]    # min max [rad/s]
            heading = [0, 0]
            height = [-0.6, 0.0]

    class asset( LeggedRobotCfg.asset ):
        file = '/home/humanoid/isaac_gym_projects/OpenHomie/HomieRL/legged_gym/resources/robots/h1_2/h1_2_27dof.urdf'
        name = "h1_2"
        foot_name = "ankle_roll"
        left_foot_name = "left_foot"
        right_foot_name = "right_foot"
        # penalize_contacts_on = ["pelvis", "torso", "shoulder", "elbow", "knee", "hip"]
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ['torso']
        # privileged_contacts_on = ["base", "thigh", "calf"]
        # curriculum_joints = ['torso_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint']
        # has all been fixed
        curriculum_joints = []

        left_leg_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint']
        right_leg_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint']
        left_hip_joints = ['left_hip_roll_joint', "left_hip_pitch_joint", "left_hip_yaw_joint"]
        right_hip_joints = ['right_hip_roll_joint', "right_hip_pitch_joint", "right_hip_yaw_joint"]
        hip_pitch_joints = ['right_hip_pitch_joint', 'left_hip_pitch_joint']

        knee_joints = ['left_knee_joint', 'right_knee_joint']
        ankle_joints = ["left_ankle_roll_joint", "right_ankle_roll_joint"]
        upper_body_link = "torso_link"
        imu_link = "imu_link"
        knee_names = ["left_knee_link", "left_hip_yaw_link", "right_knee_link", "right_hip_yaw_link"]
        self_collision = 1
        flip_visual_attachments = False
        ankle_sole_distance = 0.04 #! NOT SURE WHAT IS IDEAL!


    class domain_rand(LeggedRobotCfg.domain_rand):

        use_random = True

        randomize_joint_injection = use_random # maybe not that important, used in him each step, slow
        joint_injection_range = [-0.05, 0.05]

        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]

        randomize_payload_mass = use_random
        # payload_mass_range = [-5, 10]
        payload_mass_range = [-10, 15]  #! NOT SURE WHAT IS IDEAL!

        hand_payload_mass_range = [-0.2, 0.5]   #! NOT SURE WHAT IS IDEAL!

        randomize_com_displacement = False
        com_displacement_range = [-0.1, 0.1]

        randomize_body_displacement = use_random
        body_displacement_range = [-0.1, 0.1]

        randomize_link_mass = use_random
        link_mass_range = [0.8, 1.2]

        randomize_friction = use_random
        friction_range = [0.1, 3.0]

        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]

        randomize_kp = use_random
        kp_range = [0.9, 1.1]

        randomize_kd = use_random
        kd_range = [0.9, 1.1]

        randomize_initial_joint_pos = use_random
        initial_joint_pos_scale = [0.8, 1.2]
        initial_joint_pos_offset = [-0.1, 0.1]

        push_robots = use_random
        push_interval_s = 4
        upper_interval_s = 1
        max_push_vel_xy = 0.5

        init_upper_ratio = 0.
        delay = use_random

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # for plane work
            tracking_x_vel = 1.5
            tracking_y_vel = 1.
            tracking_ang_vel = 2.
            lin_vel_z = -0.5
            ang_vel_xy = -0.025
            orientation = -1.5#-1.25
            action_rate = -0.01
            tracking_base_height = 2. # try height tracking
            # base_height = -10.0
            # base_height_wrt_feet = 0.1
            deviation_all_joint = 0
            deviation_arm_joint = 0  #-0.1
            deviation_leg_joint = 0
            deviation_hip_joint = -0.2  #0. #-0.5
            deviation_waist_joint = 0  #-0.25
            deviation_ankle_joint = -0.5
            deviation_knee_joint = -0.75
            dof_acc = -2.5e-7
            dof_pos_limits = -2.
            feet_air_time = 0.05
            feet_clearance = -0.25  #-0.2 #-0.2
            feet_distance = 0.0
            feet_distance_lateral = 0.5  #2.5
            knee_distance_lateral = 1.0
            feet_ground_parallel = -2.0  #-2.0
            feet_parallel = -3.0  #-2.5
            smoothness = -0.05  #-0.01

            # collision = -2.5  #2.5

            joint_power = -2e-5
            feet_stumble = -1.5#-1.5  # maybe larger
            torques = -2.5e-6
            dof_vel = -1e-4
            dof_vel_limits = -2e-3
            torque_limits = -0.1
            no_fly = 0.75# just try
            joint_tracking_error = -0.1
            feet_slip = -0.25
            feet_contact_forces = -0.00025
            contact_momentum = 2.5e-4
            action_vanish = -1.0  # -1.0
            stand_still = -0.15



        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.975 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.80
        soft_torque_limit = 0.95
        base_height_target = 0.98 # default is 1.0
        max_contact_force = 600. # forces above this value are penalized
        least_feet_distance = 0.3 #! NOT SURE WHAT IS IDEAL!
        least_feet_distance_lateral = 0.3
        most_feet_distance_lateral = 0.4
        most_knee_distance_lateral = 0.4
        least_knee_distance_lateral = 0.3
        clearance_height_target = 0.2

    class env( LeggedRobotCfg.rewards ):
        num_envs = 4096
        num_actions = 12 # number of actuators on robot
        # num_dofs = 43
        num_dofs = 27
        num_one_step_observations = 2 * num_dofs + 10 + num_actions # 54 + 10 + 12 = 22 + 54 = 76
        num_one_step_privileged_obs = num_one_step_observations + 3
        num_actor_history = 6
        num_critic_history = 1
        num_observations = num_actor_history * num_one_step_observations #+ 96
        num_privileged_obs = num_critic_history * num_one_step_privileged_obs #+ 187
        action_curriculum = True
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        upper_teleop = False


    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh


    class noise( LeggedRobotCfg.terrain ):
        add_noise = True
        noise_level = 1.0
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            dof_pos = 0.02
            dof_vel = 2.0
            lin_vel = 0.1
            ang_vel = 0.5
            gravity = 0.05
            height_measurement = 0.1

class H12RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        use_flip = True # wait to determine
        entropy_coef = 0.01
        symmetry_scale = 1.0
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        save_interval = 200
        num_steps_per_env = 50
        max_iterations = 10000
        run_name = 'h1_2'
        experiment_name = 'h1_2'
        wandb_project = "h1_2"
        logger = "wandb"
        # logger = "tensorboard"
        wandb_user = ""