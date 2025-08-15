from enum import Enum

import numpy as np


class LightState(Enum):
    ON = 1
    OFF = 0


class Light:
    def __init__(self, name, cfg, uid, p, cid):
        self.name = name
        self.uid = uid
        self.p = p
        self.cid = cid
        self.link = cfg["link"]
        self.initial_state = cfg["initial_state"]
        self.link_id = next(
            i
            for i in range(self.p.getNumJoints(uid, physicsClientId=self.cid))
            if self.p.getJointInfo(uid, i, physicsClientId=self.cid)[12].decode("utf-8") == self.link
        )
        self.color_on = cfg["color"]
        self.color_off = [1, 1, 1, 1]
        self.sample_states = np.array([0, 1])
        self.state = self.sample_state()
        if self.state == LightState.ON:
            self.turn_on()
        else:
            self.turn_off()

    def reset(self, state=None):
        if state is None:
            _state = self.sample_state().value
        else:
            _state = state
        if _state == LightState.ON.value:
            self.turn_on()
        elif _state == LightState.OFF.value:
            self.turn_off()
        else:
            print("Light state can be only 0 or 1.")
            raise ValueError

    def get_state(self):
        return float(self.state.value)

    def get_pose(self, euler_obs=False):
        """Get the pose of the button link (not the base object)"""
        # Get link state for specific link
        link_state = self.p.getLinkState(self.uid, self.link_id, physicsClientId=self.cid)
        pos = link_state[0]  # World position
        orn = link_state[1]  # World orientation (quaternion)
        if euler_obs:
            orn = self.p.getEulerFromQuaternion(orn)
        return np.concatenate([pos, orn])

    def get_info(self):
        return {"logical_state": self.get_state()}

    def turn_on(self):
        self.state = LightState.ON
        self.p.changeVisualShape(self.uid, self.link_id, rgbaColor=self.color_on, physicsClientId=self.cid)

    def turn_off(self):
        self.state = LightState.OFF
        self.p.changeVisualShape(self.uid, self.link_id, rgbaColor=self.color_off, physicsClientId=self.cid)

    def serialize(self):
        return self.get_info()

    def sample_state(self):
        if isinstance(self.initial_state, str):
            if self.initial_state == "any":
                return LightState(np.random.choice(self.sample_states))
            else:
                raise ValueError(f"Invalid initial state: {self.initial_state}")
        else:
            # If initial_state is a number, return it directly
            return LightState(self.initial_state)
