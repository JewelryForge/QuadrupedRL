from burl.rl.reward import *
from burl.sim.quadruped import Quadruped, A1
import logging


class BasicTask(object):
    def __init__(self, env, cmd=(1.0, 0.0, 0.0)):
        self._env = env
        self._robot: Quadruped = env.robot
        self._rewards = {
            LinearVelocityReward(): 0.1,
            AngularVelocityReward(): 0.05,
            RedundantLinearPenalty(): 0.02,
            RedundantAngularPenalty(): 0.03,
            BodyPosturePenalty(): 0.03,
            TargetMutationPenalty(): 0.03,
            BodyHeightReward(): 0.02,
            BodyCollisionPenalty(): 0.02,
            TorquePenalty(): 0.03
        }
        self._cmd = cmd

    @property
    def cmd(self):
        return self._cmd

    def calculateReward(self):
        linear = self._robot.getBaseLinearVelocityInBaseFrame()
        angular = self._robot.getBaseAngularVelocityInBaseFrame()
        contact_states = self._robot.getContactStates()

        mutation = self._env.getActionMutation()
        x, y, z = self._robot.getBasePosition(False)
        body_height = z - self._env.getTerrainHeight(x, y)
        torques = self._robot.getLastAppliedTorques()
        orientation = self._robot.orientation
        args = (
            (self._cmd, linear),  # Linear Rew
            (self._cmd, angular),  # Angular Rew
            (self._cmd, linear),  # Linear Pen
            (angular,),  # Angular Pen
            (orientation,),  # Posture Pen
            (mutation,),  # Target Mut Pen
            (body_height,),  # Height Rew
            (contact_states,),  # Collision Pen
            (torques,)  # Torque Pen
        )
        rewards = [r(*arg) for r, arg in zip(self._rewards, args)]
        weighted_rewards = [r * w for r, w in zip(rewards, self._rewards.values())]
        # print(np.array(linear), np.array(angular))
        # print(Rpy.from_quaternion(orientation))
        # print(np.array(rewards))
        # print(np.array(weighted_rewards))
        # print()
        # logging.debug('VEL:', str(np.array(linear)), str(np.array(angular)))
        # logging.debug('REW:', str(np.array(rewards)))
        # logging.debug('WEIGHTED:', str(np.array(weighted_rewards)))
        return sum(weighted_rewards)

    def reset(self):
        pass

    MAX_MOVEMENT_ANGLE = 1.0

    def done(self):
        joint_diff = self._robot.getJointPositions(noisy=False) - self._robot.STANCE_POSTURE
        if any(joint_diff > self.MAX_MOVEMENT_ANGLE) or \
                any(joint_diff < -self.MAX_MOVEMENT_ANGLE):
            return True
        return False
