import collections
import time

import numpy as np
import pybullet
import pybullet_data
from pybullet_utils import bullet_client
from collections import namedtuple

JointState = namedtuple('JointState', ('pos', 'vel', 'reaction_force', 'torque'),
                        defaults=(0., 0., (0., 0., 0., 0., 0., 0.), 0.))
ObservationRaw = namedtuple('ObservationRaw', ('joint_states', 'base_state', 'contact_states'))


class Quadruped:
    INIT_POSITION = [0, 0, .6]
    INIT_RACK_POSITION = [0, 0, 1]
    INIT_ORIENTATION = [0, 0, 0, 1]
    NUM_MOTORS = 12
    LEG_NAMES = ['FL', 'RL', 'RR', 'FR']
    JOINT_TYPES = ['hip', 'thigh', 'calf', 'foot']
    JOINT_SUFFIX = {'hip': 'joint', 'thigh': 'joint', 'calf': 'joint', 'foot': 'fixed'}

    def __init__(self, **kwargs):
        self.urdf_file: str = kwargs.get('urdf_file')
        self._bullet = kwargs.get('pybullet_client', pybullet)
        self._time_step: float = kwargs.get('time_step', 1 / 240)
        self._self_collision_enabled: bool = kwargs.get('self_collision_enabled', False)
        self._render_enabled: bool = kwargs.get('_render_enabled', True)
        self._latency: float = kwargs.get('latency', 0)
        assert self._latency >= 0 and self._time_step > 0

        self._bullet.setTimeStep(self._time_step)
        self._joint_states: list[JointState] = []
        self._base_position: np.ndarray = np.zeros(3)
        self._base_orientation: np.ndarray = np.zeros(4)
        self._contact_states: np.ndarray = np.zeros(12)
        self._step_counter: int = 0
        # self._foot_contact_states: np.ndarray = np.zeros(4, bool)
        # self._thigh_contact_states: np.ndarray = np.zeros(4, bool)
        # self._shank_contact_states: np.ndarray = np.zeros(4, bool)

        self._latency_steps = int(self._latency // self._time_step)
        self.quadruped: int = self._load_robot_urdf()
        self._joint_names = ['floating_base']
        self._joint_names += ['_'.join((l, j, self.JOINT_SUFFIX[j])) for l in self.LEG_NAMES for j in self.JOINT_TYPES]
        self._joint_ids: list = self._get_joint_ids()
        self._joint_id_dict = {'base': self._joint_ids[0]}
        self._joint_id_dict.update(dict(zip(
            [(l, j) for l in self.LEG_NAMES for j in self.JOINT_TYPES], self._joint_ids[1:]
        )))
        self._observation_history = collections.deque(maxlen=1000)

        for leg in self.LEG_NAMES:
            self._bullet.enableJointForceTorqueSensor(self.quadruped, self._joint_id_dict[(leg, 'foot')])

    def _get_joint_ids(self):
        joint_name_to_id = {}
        for i in range(self._bullet.getNumJoints(self.quadruped)):
            joint_info = self._bullet.getJointInfo(self.quadruped, i)
            joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
        return [joint_name_to_id.get(n, -1) for n in self._joint_names]

    def ik(self, leg):
        pass

    def fk(self, leg):
        pass

    def _load_robot_urdf(self, pos=None, orn=None):
        if self._self_collision_enabled:
            return self._bullet.loadURDF(
                self.urdf_file,
                pos if pos else self.INIT_POSITION,
                orn if orn else self.INIT_ORIENTATION,
                flags=self._bullet.URDF_USE_SELF_COLLISION)
        else:
            return self._bullet.loadURDF(
                self.urdf_file,
                pos if pos else self.INIT_POSITION,
                orn if orn else self.INIT_ORIENTATION)

    def _get_observation(self):
        if len(self._observation_history) <= self._latency_steps:
            return ObservationRaw(tuple(JointState() for _ in range(len(self._joint_ids))),
                                  ((0., 0., 0.), (0., 0., 0., 0.)),
                                  tuple(False for _ in range(self.NUM_MOTORS + 1)))

        return self._observation_history[-(self._latency_steps + 1)]

    def _get_base_position(self, noisy=False):
        return self._add_sensor_noise(self._base_position) if noisy else self._base_position

    def _get_base_rpy(self, noisy=False):
        rpy = self._bullet.getEulerFromQuaternion(self._base_orientation)
        return self._add_sensor_noise(rpy) if noisy else rpy

    def _reset_at_current_state(self):
        x, y, _ = self._get_base_position()
        z = self.INIT_POSITION[2]
        _, _, yaw = self._get_base_rpy()
        orn_q = self._bullet.getQuaternionFromEuler([0.0, 0.0, yaw])
        self._bullet.resetBasePositionAndOrientation(self.quadruped, [x, y, z], orn_q)
        self._bullet.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])

    def _add_sensor_noise(self, data):
        return data  # TODO

    def step(self):
        self._bullet.stepSimulation()
        self.update_observation()

    def update_observation(self):
        self._step_counter += 1
        joint_states = [JointState(*js) for js in self._bullet.getJointStates(self.quadruped, self._joint_ids)]
        position, orientation = self._bullet.getBasePositionAndOrientation(self.quadruped)
        contact_states = self._get_contact_states()
        self._observation_history.append(ObservationRaw(joint_states, (position, orientation), contact_states))
        observation = self._get_observation()
        self._joint_states = observation.joint_states
        self._base_position, self._base_orientation = observation.base_state
        self._contact_states = observation.contact_states

    def _get_contact_state(self, link_id):
        return self._bullet.getContactPoints(bodyA=0, linkIndexA=-1, bodyB=self.quadruped, linkIndexB=link_id)

    # cnt = 0

    def _get_contact_states(self):
        base_contact = bool(self._get_contact_state(self._joint_id_dict['base']))
        contact_states = [base_contact]
        # self.cnt += 1
        # print(int(base_contact), end=' ')
        for leg in self.LEG_NAMES:
            thigh_contact = bool(self._get_contact_state(self._joint_id_dict[(leg, 'hip')])) or \
                            bool(self._get_contact_state(self._joint_id_dict[(leg, 'thigh')]))
            shank_contact = bool(self._get_contact_state(self._joint_id_dict[(leg, 'calf')]))
            foot_contact = bool(self._get_contact_state(self._joint_id_dict[(leg, 'foot')]))
            contact_states.extend((thigh_contact, shank_contact, foot_contact))

        #     c1 = bool(self._get_contact_state(self._joint_id_dict[(leg, 'hip')]))
        #     c2 = bool(self._get_contact_state(self._joint_id_dict[(leg, 'thigh')]))
        #     c3 = bool(self._get_contact_state(self._joint_id_dict[(leg, 'calf')]))
        #     c4 = bool(self._get_contact_state(self._joint_id_dict[(leg, 'foot')]))
        #     print(*[int(c) for c in [c1, c2, c3, c4]], sep='', end=' ')
        # print()
        return contact_states


if __name__ == '__main__':
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -9.8)
    planeId = p.loadURDF("plane.urdf")
    q = Quadruped(urdf_file="/home/jewel/Workspaces/teacher-student/urdf/aliengo/xacro/aliengo.urdf",
                  pybullet_client=p)
    # c = p.loadURDF("cube.urdf")

    for _ in range(100000):
        q.step()
        # q.get_foot_contacts()
        time.sleep(1. / 240.)
