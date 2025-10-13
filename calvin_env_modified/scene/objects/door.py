import numpy as np


MAX_FORCE = 4


class Door:
    def __init__(self, name, cfg, uid, p, cid):
        self.name = name
        self.p = p
        self.cid = cid
        # get joint_index by name (to prevent index errors when additional joints are added)
        joint_index = next(
            i
            for i in range(self.p.getNumJoints(uid, physicsClientId=self.cid))
            if self.p.getJointInfo(uid, i, physicsClientId=self.cid)[1].decode("utf-8") == name
        )
        self.joint_index = joint_index
        self.link = cfg["link"]
        self.link_id = next(
            i
            for i in range(self.p.getNumJoints(uid, physicsClientId=self.cid))
            if self.p.getJointInfo(uid, i, physicsClientId=self.cid)[12].decode("utf-8") == self.link
        )
        self.uid = uid
        self.initial_state = cfg["initial_state"]
        self.p.setJointMotorControl2(
            self.uid,
            self.joint_index,
            controlMode=p.VELOCITY_CONTROL,
            force=MAX_FORCE,
            physicsClientId=self.cid,
        )
        self.ll, self.ul = self.p.getJointInfo(uid, joint_index, physicsClientId=self.cid)[8:10]
        self.sample_states = np.array([self.ll, self.ul])

    def reset(self, state=None):
        _state = self.sample_state() if state is None else state
        self.p.resetJointState(
            self.uid,
            self.joint_index,
            _state,
            physicsClientId=self.cid,
        )

    def get_state(self):
        joint_state = self.p.getJointState(self.uid, self.joint_index, physicsClientId=self.cid)
        return float(joint_state[0])

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
        return {"current_state": self.get_state()}
    
    def sample_state(self):
        if isinstance(self.initial_state, str):
            if self.initial_state == "any":
                return np.random.choice(self.sample_states)
            else:
                raise ValueError(f"Invalid initial state: {self.initial_state}")
        else:
            # If initial_state is a number, return it directly
            return self.initial_state
