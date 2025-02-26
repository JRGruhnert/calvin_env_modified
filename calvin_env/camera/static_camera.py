import numpy as np
import pybullet as p

from calvin_env.camera.camera import Camera


class StaticCamera(Camera):
    def __init__(
        self,
        fov,
        aspect,
        nearval,
        farval,
        width,
        height,
        look_at,
        look_from,
        up_vector,
        cid,
        name,
        robot_id=None,
        objects=None,
    ):
        """
        Initialize the camera
        Args:
            argument_group: initialize the camera and add needed arguments to argparse

        Returns:
            None
        """
        self._nearval = nearval
        self._farval = farval
        self.fov = fov
        self.aspect = aspect
        self.look_from = look_from
        self.look_at = look_at
        self.up_vector = up_vector
        self.width = width
        self.height = height
        self._viewMatrix = p.computeViewMatrix(
            cameraEyePosition=look_from, cameraTargetPosition=look_at, cameraUpVector=self.up_vector
        )
        self._projectionMatrix = p.computeProjectionMatrixFOV(
            fov=fov, aspect=aspect, nearVal=self._nearval, farVal=self._farval
        )
        self.cid = cid
        self._name = name

    def set_position_from_gui(self):
        info = p.getDebugVisualizerCamera(physicsClientId=self.cid)
        look_at = np.array(info[-1])
        dist = info[-2]
        forward = np.array(info[5])
        look_from = look_at - dist * forward
        self._viewMatrix = p.computeViewMatrix(
            cameraEyePosition=look_from, cameraTargetPosition=look_at, cameraUpVector=self.up_vector
        )
        look_from = [float(x) for x in look_from]
        look_at = [float(x) for x in look_at]
        return look_from, look_at

    def _render(self):
        return p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self._viewMatrix,
            projectionMatrix=self._projectionMatrix,
            physicsClientId=self.cid,
        )
