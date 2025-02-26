import logging
from math import pi
import os
from pathlib import Path
import pickle
import pkgutil
import re
import sys
import time
from loguru import logger
import cv2
import gym
import gym.utils
import gym.utils.seeding
import hydra
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import torch

import calvin_env
from calvin_env.camera.camera import Camera
from calvin_env.robot.robot import Robot
from calvin_env.utils.utils import FpsController, get_git_commit_hash
from tapas_gmm.utils.observation import dict_to_tensordict

# A logger for this file
log = logging.getLogger(__name__)

import numpy as np


from typing import Tuple

import numpy as np


class CalvinObservation(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(
        self,
        wrist_rgb: np.ndarray,
        wrist_depth: np.ndarray,
        wrist_mask: np.ndarray,
        wrist_point_cloud: np.ndarray,
        front_rgb: np.ndarray,
        front_depth: np.ndarray,
        front_mask: np.ndarray,
        front_point_cloud: np.ndarray,
        joint_velocities: np.ndarray,
        joint_positions: np.ndarray,
        joint_forces: np.ndarray,
        gripper_open: float,
        gripper_pose: np.ndarray,
        gripper_matrix: np.ndarray,
        gripper_joint_positions: np.ndarray,
        gripper_touch_forces: np.ndarray,
        task_low_dim_state: np.ndarray,
        misc: dict,
    ):
        self.wrist_rgb = wrist_rgb
        self.wrist_depth = wrist_depth
        self.wrist_mask = wrist_mask
        self.wrist_point_cloud = wrist_point_cloud
        self.front_rgb = front_rgb
        self.front_depth = front_depth
        self.front_mask = front_mask
        self.front_point_cloud = front_point_cloud
        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        self.joint_forces = joint_forces
        self.gripper_open = gripper_open
        self.gripper_pose = gripper_pose
        self.gripper_matrix = gripper_matrix
        self.gripper_joint_positions = gripper_joint_positions
        self.gripper_touch_forces = gripper_touch_forces
        self.task_low_dim_state = task_low_dim_state
        self.misc = misc

    def get_low_dim_data(self) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional obseervations.

        :return: 1D array of observations.
        """
        low_dim_data = [] if self.gripper_open is None else [[self.gripper_open]]
        for data in [
            self.joint_velocities,
            self.joint_positions,
            self.joint_forces,
            self.gripper_pose,
            self.gripper_joint_positions,
            self.gripper_touch_forces,
            self.task_low_dim_state,
        ]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])


class CalvinEnv(gym.Env):
    def __init__(
        self,
        robot_cfg,
        seed,
        use_vr,
        bullet_time_step,
        cameras,
        show_gui,
        scene_cfg,
        use_scene_info,
        use_egl,
        control_freq=10,
    ):
        self.p = p
        # for calculation of FPS
        self.t = time.time()
        self.prev_time = time.time()
        self.fps_controller = FpsController(bullet_time_step)
        self.use_vr = use_vr
        self.show_gui = show_gui
        self.use_scene_info = use_scene_info
        self.cid = -1
        self.ownsPhysicsClient = False
        self.use_egl = use_egl
        self.control_freq = control_freq
        self.action_repeat = int(bullet_time_step // control_freq)
        render_width = max([cameras[cam].width for cam in cameras]) if cameras else None
        render_height = max([cameras[cam].height for cam in cameras]) if cameras else None
        self.initialize_bullet(bullet_time_step, render_width, render_height)
        self.np_random = None
        self.seed(seed)
        self.robot = hydra.utils.instantiate(robot_cfg, cid=self.cid)
        self.scene = hydra.utils.instantiate(scene_cfg, p=self.p, cid=self.cid, np_random=self.np_random)

        # Load Env
        self.load()

        # init cameras after scene is loaded to have robot id available
        self.cameras: list[Camera] = [
            hydra.utils.instantiate(
                cameras[name], cid=self.cid, robot_id=self.robot.robot_uid, objects=self.scene.get_objects()
            )
            for name in cameras
        ]
        log.info(f"Using calvin_env with commit {get_git_commit_hash(Path(calvin_env.__file__))}.")

    def __del__(self):
        self.close()

    def reset(self, robot_obs=None, scene_obs=None):
        self.scene.reset(scene_obs)
        self.robot.reset(robot_obs)
        self.p.stepSimulation(physicsClientId=self.cid)
        return self._get_observation()

    # From pybullet gym_manipulator_envs code
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/gym_manipulator_envs.py
    def initialize_bullet(self, bullet_time_step, render_width, render_height):
        if self.cid < 0:
            self.ownsPhysicsClient = True
            if self.use_vr:
                self.p = bc.BulletClient(connection_mode=p.SHARED_MEMORY)
                cid = self.p._client
                if cid < 0:
                    log.error("Failed to connect to SHARED_MEMORY bullet server.\n" " Is it running?")
                    sys.exit(1)
                self.p.setRealTimeSimulation(enableRealTimeSimulation=1, physicsClientId=cid)
            elif self.show_gui:
                self.p = bc.BulletClient(connection_mode=p.GUI)
                cid = self.p._client
                if cid < 0:
                    log.error("Failed to connect to GUI.")
                self.p.resetDebugVisualizerCamera(
                    cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0]
                )
            elif self.use_egl:
                options = f"--width={render_width} --height={render_height}"
                self.p = p
                cid = self.p.connect(p.DIRECT, options=options)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
                p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=cid)
                p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=cid)
                p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=cid)
                egl = pkgutil.get_loader("eglRenderer")
                log.info("Loading EGL plugin (may segfault on misconfigured systems)...")
                if egl:
                    plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                else:
                    plugin = p.loadPlugin("eglRendererPlugin")
                if plugin < 0:
                    log.error("\nPlugin Failed to load!\n")
                    sys.exit()
                # set environment variable for tacto renderer
                os.environ["PYOPENGL_PLATFORM"] = "egl"
                log.info("Successfully loaded egl plugin")
            else:
                self.p = bc.BulletClient(connection_mode=p.DIRECT)
                cid = self.p._client
                if cid < 0:
                    log.error("Failed to start DIRECT bullet mode.")
            log.info(f"Connected to server with id: {cid}")

            self.cid = cid
            self.p.resetSimulation(physicsClientId=self.cid)
            self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=self.cid)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
            log.info(f"Connected to server with id: {self.cid}")
            self.p.setTimeStep(1.0 / bullet_time_step, physicsClientId=self.cid)
            return cid

    def load(self):
        log.info("Resetting simulation")
        self.p.resetSimulation(physicsClientId=self.cid)
        log.info("Setting gravity")
        self.p.setGravity(0, 0, -9.8, physicsClientId=self.cid)

        self.robot.load()
        self.scene.load()

    def close(self):
        if self.ownsPhysicsClient:
            print("disconnecting id %d from server" % self.cid)
            if self.cid >= 0 and self.p is not None:
                try:
                    self.p.disconnect(physicsClientId=self.cid)
                except TypeError:
                    pass

        else:
            print("does not own physics client id")

    def render(self, mode="human"):
        """render is gym compatibility function"""
        obs: CalvinObservation = self._get_observation()

        if mode == "human":
            # Resize images to the desired size
            rgb_static = cv2.resize(obs.front_rgb[:, :, ::-1], (500, 500))
            depth_static = cv2.resize(obs.front_depth, (500, 500))
            rgb_gripper = cv2.resize(obs.wrist_rgb[:, :, ::-1], (500, 500))
            depth_gripper = cv2.resize(obs.wrist_depth, (500, 500))

            # Convert depth images to BGR if they are single-channel
            depth_static_color = cv2.cvtColor(depth_static, cv2.COLOR_GRAY2BGR)
            depth_gripper_color = cv2.cvtColor(depth_gripper, cv2.COLOR_GRAY2BGR)

            # Create a 2x2 grid of images
            top_row = np.hstack((rgb_static, depth_static_color))
            bottom_row = np.hstack((rgb_gripper, depth_gripper_color))
            combined = np.vstack((top_row, bottom_row))

            # Display the combined image
            cv2.imshow("Combined View", combined)
            cv2.waitKey(1)
        else:
            raise NotImplementedError

    def reset(self, robot_obs=None, scene_obs=None):
        self.scene.reset(scene_obs)
        self.robot.reset(robot_obs)
        self.p.stepSimulation(physicsClientId=self.cid)
        return self._get_observation(), self._get_info()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        # self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def _get_observation(
        self,
        has_joint_forces=True,
        has_gripper_touch_forces=True,
    ) -> CalvinObservation:

        camera_map = {}
        for cam in self.cameras:
            camera_map[cam.name] = cam
        gripper_rgb, gripper_depth, gripper_pcd, gripper_mask = camera_map["gripper"].render()
        static_rgb, static_depth, static_pcd, static_mask = camera_map["static"].render()

        _, robot_obs = self.robot.get_observation()  # get state observation
        scene_obs = self.scene.get_low_dim_state()
        arm_joint_forces = robot_obs["arm_joint_forces"]
        arm_joint_velocities = robot_obs["arm_joint_velocities"]
        arm_joint_positions = robot_obs["arm_joint_positions"]

        joint_forces = None
        if has_joint_forces:
            joint_forces = self.robot.joint_forces_noise.apply(
                np.array([-f if v < 0 else f for f, v in zip(arm_joint_forces, arm_joint_velocities)])
            )

        ee_forces_flat = None
        if has_gripper_touch_forces:
            ee_forces = robot_obs["gripper_finger_forces"]
            ee_forces_flat = []
            for eef in ee_forces:
                ee_forces_flat.extend(eef)
            ee_forces_flat = np.array(ee_forces_flat)

        low_dim_state_tensor = dict_to_tensordict(
            {f"obj{i:03d}": torch.Tensor(state) for i, state in enumerate(scene_obs)},
            # batch_size=empty_batchsize,
        )

        print(f"Low Dim State Tensor: {low_dim_state_tensor}")

        obs = CalvinObservation(
            wrist_rgb=gripper_rgb,
            wrist_depth=gripper_depth,
            wrist_point_cloud=gripper_pcd,
            front_rgb=static_rgb,
            front_depth=static_depth,
            front_point_cloud=static_pcd,
            wrist_mask=gripper_mask,
            front_mask=static_mask,
            joint_velocities=self.robot.joint_velocities_noise.apply(np.array(arm_joint_velocities)),
            joint_positions=self.robot.joint_positions_noise.apply(np.array(arm_joint_positions)),
            joint_forces=joint_forces,
            gripper_open=robot_obs["gripper_opening_state"],
            gripper_matrix=robot_obs["gripper_view_matrix"],
            gripper_pose=robot_obs["gripper_pose"],
            gripper_touch_forces=ee_forces_flat,
            gripper_joint_positions=robot_obs["gripper_finger_positions"],
            task_low_dim_state=self.scene.get_low_dim_state(),
            misc=self._get_misc(),
        )
        return obs

    def _get_state_obs(self):
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
        """
        robot_obs, robot_info = self.robot.get_observation()
        scene_obs = self.scene.get_obs()
        obs = {"robot_obs": robot_obs, "scene_obs": scene_obs, "robot_info": robot_info}
        return obs

    def _get_info(self):
        _, robot_info = self.robot.get_observation()
        info = {"robot_info": robot_info}
        if self.use_scene_info:
            info["scene_info"] = self.scene.get_info()
        return info

    def _get_misc(self):
        def _get_cam_data(cam: Camera, name: str):
            d = {
                "%s_extrinsics" % name: cam.viewMatrix,
                "%s_intrinsics" % name: cam.projectionMatrix,
                "%s_near" % name: cam.nearval,
                "%s_far" % name: cam.farval,
            }
            return d

        camera_dict = {}
        for cam in self.cameras:
            camera_dict[cam.name] = cam

        misc = _get_cam_data(camera_dict["static"], "front_camera")
        misc.update(_get_cam_data(camera_dict["gripper"], "wrist_camera"))
        misc.update({"variation_index": self.seed()})

        # if self._joint_position_action is not None:
        # Store the actual requested joint positions during demo collection
        #    misc.update({"joint_position_action": self._joint_position_action})
        # joint_poses = [j.get_pose() for j in self.robot.arm.joints]
        # misc.update({"joint_poses": joint_poses})
        return misc

    def step(self, action):
        self.robot.apply_action(action)
        for i in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)
        self.scene.step()
        obs = self._get_observation()
        info = self._get_info()
        # obs, reward, done, info
        return obs, 0, False, info

    def reset_from_storage(self, filename):
        """
        Args:
            filename: file to load from.
        Returns:
            observation
        """
        with open(filename, "rb") as file:
            data = pickle.load(file)

        self.robot.reset_from_storage(data["robot"])
        self.scene.reset_from_storage(data["scene"])

        self.p.stepSimulation(physicsClientId=self.cid)

        return data["state_obs"], data["done"], data["info"]

    def serialize(self):
        data = {"time": time.time_ns() / (10**9), "robot": self.robot.serialize(), "scene": self.scene.serialize()}
        return data


def get_env(dataset_path, obs_space=None, show_gui=True, **kwargs):
    from pathlib import Path

    from omegaconf import OmegaConf

    render_conf = OmegaConf.load(Path(dataset_path) / ".hydra" / "merged_config.yaml")

    if obs_space is not None:
        exclude_keys = set(render_conf.cameras.keys()) - {
            re.split("_", key)[1] for key in obs_space["rgb_obs"] + obs_space["depth_obs"]
        }
        for k in exclude_keys:
            del render_conf.cameras[k]
    if "scene" in kwargs:
        scene_cfg = OmegaConf.load(Path(calvin_env.__file__).parents[1] / "conf/scene" / f"{kwargs['scene']}.yaml")
        render_conf.scene = scene_cfg
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    env = hydra.utils.instantiate(render_conf.env, show_gui=show_gui, use_vr=False, use_scene_info=True)
    return env


@hydra.main(config_path="../../conf", config_name="config_motion_data_collection")
def run_env(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=True, use_vr=False, use_scene_info=True)

    env.reset()
    while True:
        action = {"action": np.array((0.0, 0, 0, 0, 0, 0, 1)), "type": "cartesian_rel"}
        # cartesian actions can also be input directly as numpy arrays
        # action = np.array((0., 0, 0, 0, 0, 0, 1))

        # relative action in joint space
        # action = {"action": np.array((0., 0, 0, 0, 0, 0, 0, 1)),
        #           "type": "joint_rel"}

        env.step(action)
        # env.render()
        time.sleep(0.01)


def get_env_from_cfg():
    """Bypass Hydra's execution context and create the environment manually."""
    with hydra.initialize(config_path="../../conf"):
        cfg = hydra.compose(config_name="config_motion_data_collection")
        env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)

        env = CalvinEnv(
            robot_cfg=cfg.robot,  # Robot basic cfg load here
            seed=cfg.seed,  # ignore for now
            use_vr=False,  # correct
            bullet_time_step=cfg.bullet_time_step,  # ignore for now
            cameras=cfg.cameras,
            show_gui=False,
            scene_cfg=cfg.scene,
            use_scene_info=True,
            use_egl=False,
        )
        assert env is not None, "Failed to create CustomSimEnv"
        return env


if __name__ == "__main__":
    run_env()
