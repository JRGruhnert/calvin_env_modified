import collections
import logging

import numpy as np
import pybullet as p
import quaternion  # noqa

import calvin_env.utils.utils as utils

from termcolor import cprint

from pybullet_planning import BASE_LINK, RED, BLUE, GREEN
import pybullet_planning as planner


# A logger for this file
log = logging.getLogger(__name__)

class PandaArmMotionPlanningSolver:
    """
    pyBullet MotionPlanner for the Panda Arm
    """

    def __init__(self, env, limit_angle, visualize_vr_pos, reset_button_queue_len):
        planner.connect(use_gui=True)

        # * zoom in so we can see it, this is optional
        camera_base_pt = (0,0,0)
        camera_pt = np.array(camera_base_pt) + np.array([0.1, -0.1, 0.1])
        planner.set_camera_pose(tuple(camera_pt), camera_base_pt)
        
        self.visualize_vr_pos = visualize_vr_pos
        self.vr_pos_uid = None
        if visualize_vr_pos:
            self.vr_pos_uid = self.create_vr_pos_visualization_shape()
        log.info("Disable Picking")
        p.configureDebugVisualizer(p.COV_ENABLE_VR_PICKING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_VR_RENDER_CONTROLLERS, 0)
        self._prev_vr_events = None
        self.prev_action = None

        # wait until first vr action event arrives
        while self.prev_action is None:
            _ = self.get_vr_action()

    def create_route(self, start, goal, **kwargs):
        """
        Create a route from start to goal
        """

        route  = planner.lazy_prm(start, goal, **kwargs)
        return planner.plan_joint_motion(start, goal, **kwargs)
    
    def step(self, state_obs):
        """
        Collect state observation dict
        --state_obs
            --robot_obs
                --robot_state_full
                    -- [tcp_pos, tcp_orn, gripper_opening_width]
                --gripper_opening_width
                --arm_joint_states
                --gripper_action}
            --scene_obs
        
        robot_obs, robot_info = self.robot.get_observation()
        scene_obs = self.scene.get_obs()
        obs = {"robot_obs": robot_obs, "scene_obs": scene_obs}
        """
        robot_obs = state_obs["robot_obs"]
        scene_obs = state_obs["scene_obs"]
        # * get the current end effector position
        tcp_pos, tcp_orn, gripper_opening_width = robot_obs["robot_state_full"]
        # * get the current arm joint states
        arm_joint_states = robot_obs["arm_joint_states"]
        # * get the current gripper action
        gripper_action = robot_obs["gripper_action"]
        # * get the current gripper opening width
        gripper_opening_width = robot_obs["gripper_opening_width"]

        self.prev_action = desired_ee_pos, desired_ee_orn, gripper_action
        return desired_ee_pos, desired_ee_orn, gripper_action


