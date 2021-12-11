from burl.rl.reward import *
from burl.utils import g_cfg


class BasicTask(object):
    def __init__(self, env, cmd=(1.0, 0.0, 0.0)):
        self._env = env
        self._cmd = np.asarray(cmd)
        from burl.rl.curriculum import BasicTerrainManager
        self._terrain: BasicTerrainManager = None
        self._rewards, self._weights = [], []
        for r, w in self.rewards_weights:
            self._rewards.append(r)
            self._weights.append(w)
        self._weights = np.array(self._weights)
        # self._weights = self._weights * 0.1 / self._weights.sum()
        self._weights = self._weights * 0.25
        self._details = {}

    @property
    def cmd(self):
        return self._cmd

    @property
    def robot(self):
        return self._env.robot

    rewards_weights = (
        # (EluLinearVelocityReward(), 0.1),
        (LinearVelocityReward(), 0.1),
        (YawRateReward(), 0.08),
        # (AngularVelocityReward(), 0.1),
        (BodyHeightReward(), 0.05),
        (RedundantLinearPenalty(), 0.03),
        # (RedundantAngularPenalty(), 0.03),
        (RollPitchRatePenalty(), 0.03),
        (BodyPosturePenalty(), 0.03),
        # (FootSlipPenalty(), 0.02),
        (SmallStridePenalty(), 0.2),
        # (TargetMutationPenalty(), 0.02),
        (BodyCollisionPenalty(), 0.02),
        # (TorquePenalty(), 0.01)
        (CostOfTransportReward(), 0.035)
    )

    def calculateReward(self):
        linear = self.robot.getBaseLinearVelocityInBaseFrame()
        # angular = self.robot.getBaseAngularVelocityInBaseFrame()
        r_rate, p_rate, y_rate = self.robot.getBaseRpyRate()
        body_height = self._env.getSafetyHeightOfRobot()
        safety_r, safety_p, _ = self._env.getSafetyRpyOfRobot()
        # print(body_rpy, self.robot.rpy)
        contact_states = self.robot.getContactStates()
        # mutation = self._env.getActionMutation()
        # slip = np.sum(self.robot.getFootSlipVelocity())
        strides = np.array([np.dot(s, self._cmd[:2]) for s in self.robot.getStrides()])
        # torques = self.robot.getLastAppliedTorques()
        cot = self.robot.getCostOfTransport()
        args = (
            (self._cmd, linear),  # Linear Rew
            # (self._cmd, angular),  # Angular Rew
            (self._cmd[2], y_rate),  # yaw rate Rew
            (body_height,),  # Height Rew
            (self._cmd, linear),  # Linear Pen
            # (angular,),  # Angular Pen
            (r_rate, p_rate),  # rp rate Pen
            (safety_r, safety_p),  # Posture Pen
            # (slip,),  # Slip Pen
            (strides,),  # Small Stride Pen
            # (mutation,),  # Target Mut Pen
            (contact_states,),  # Collision Pen
            # (torques,)  # Torque Pen
            (cot,)  # COT Rew
        )

        assert len(args) == len(self._rewards)
        rewards = [r(*arg) for r, arg in zip(self._rewards, args)]
        self._details = dict(zip((r.__class__.__name__ for r in self._rewards), rewards))
        weighted_sum = sum([r * w for r, w in zip(rewards, self._weights)])
        self.sendOrPrint(locals())
        return weighted_sum

    def sendOrPrint(self, variables: dict):
        def get(key):
            return variables[key]
        # print(self.robot.getCostOfTransport())
        # print(get('y_rate'), YawRateReward()(0, get('y_rate')))
        # if any(strides := get('strides')):
        #     print(strides, SmallStridePenalty()(strides))
        # if strides[0]:
        #     print()
        # print(get('r_rate'), get('p_rate'), get('y_rate'))
        # print('slip', get('slip'))
        # print()
        # from burl.utils.transforms import Rpy
        # from burl.utils import udp_pub
        # print(self.robot.getFootSlipVelocity())
        # if any(strides != 0.0)
        #     print(strides, sum(self.reshape(s) for s in strides if s != 0.0))
        # locals().update(variables)
        # data = {'AngVel': dict(zip('xyz', self.robot.getBaseAngularVelocity())),
        #         'dEuler': dict(zip(('dr', 'dp', 'dy'), self.robot.getBaseRpyRate()))}
        # udp_pub.send(data)
        # print(strides, self.robot.getStrides())
        pass

    def getRewardDetails(self):
        return self._details

    def reset(self):
        pass

    def makeTerrain(self):
        from burl.rl.curriculum import (PlainTerrainManager, TerrainCurriculum,
                                        FixedRoughTerrainManager, SlopeTerrainManager)
        if g_cfg.trn_type == 'plain':
            self._terrain = PlainTerrainManager(self._env.client)
        elif g_cfg.trn_type == 'curriculum':
            g_cfg.trn_offset = tuple(g_cfg.trn_size / 6 * self._cmd)
            self._terrain = TerrainCurriculum(self._env.client)
        elif g_cfg.trn_type == 'rough':
            self._terrain = FixedRoughTerrainManager(self._env.client, seed=2)
        elif g_cfg.trn_type == 'slope':
            self._terrain = SlopeTerrainManager(self._env.client)
        else:
            raise RuntimeError(f'Unknown terrain type {g_cfg.trn_type}')
        return self._terrain

    def register(self, episode_len):
        distance = np.dot(self.robot.position, self._cmd)
        return self._terrain.register(episode_len, distance)

    def isFailed(self):  # TODO: CHANGE TIME_OUT TO NORMALLY FINISH
        rob, env = self.robot, self._env
        r, _, _ = rob.rpy
        safety_h = env.getSafetyHeightOfRobot()
        h_lb, h_ub = g_cfg.safe_height_range
        if ((safety_h < h_lb or safety_h > h_ub) or
                (r < -np.pi / 3 or r > np.pi / 3) or
                rob.getBaseContactState()):
            return True
        joint_diff = rob.getJointPositions() - rob.STANCE_POSTURE
        if any(joint_diff > g_cfg.joint_angle_range) or any(joint_diff < -g_cfg.joint_angle_range):
            return True
        return False


class RandomCmdTask(BasicTask):
    def __init__(self, env, seed=None):
        np.random.seed(seed)
        angle = np.random.random() * 2 * np.pi
        super().__init__(env, (np.cos(angle), np.sin(angle), 0))

    def reset(self):
        angle = np.random.random() * 2 * np.pi
        self._cmd = (np.cos(angle), np.sin(angle), 0)
