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
        (BodyHeightReward(), 0.05),
        (RedundantLinearPenalty(), 0.04),
        (RollPitchRatePenalty(), 0.04),
        (BodyPosturePenalty(), 0.04),
        (FootSlipPenalty(), 0.04),
        (SmallStridePenalty(), 0.08),
        # (TargetMutationPenalty(), 0.02),
        (BodyCollisionPenalty(), 0.02),
        (CostOfTransportReward(), 0.04)
    )

    def calculateReward(self):
        linear = self.robot.getBaseLinearVelocityInBaseFrame()
        r_rate, p_rate, y_rate = self.robot.getBaseRpyRate()
        body_height = self._env.getSafetyHeightOfRobot()
        safety_r, safety_p, _ = self._env.getSafetyRpyOfRobot()
        # print(body_rpy, self.robot.rpy)
        contact_states = self.robot.getContactStates()
        # mutation = self._env.getActionMutation()
        slips = self.robot.getFootSlipVelocity()
        strides = np.array([np.dot(s, self._cmd[:2]) for s in self.robot.getStrides()])
        # torques = self.robot.getLastAppliedTorques()
        cot = self.robot.getCostOfTransport()
        args = (
            (self._cmd, linear),  # Linear Rew
            (self._cmd[2], y_rate),  # yaw rate Rew
            (body_height,),  # Height Rew
            (self._cmd, linear),  # Linear Pen
            (r_rate, p_rate),  # rp rate Pen
            (safety_r, safety_p),  # Posture Pen
            (slips,),  # Slip Pen
            (strides,),  # Small Stride Pen
            # (mutation,),  # Target Mut Pen
            (contact_states,),  # Collision Pen
            (cot,)  # COT Rew
        )

        assert len(args) == len(self._rewards)
        rewards = [r(*arg) for r, arg in zip(self._rewards, args)]
        self._details = dict(zip((r.__class__.__name__ for r in self._rewards), rewards))
        weighted_sum = sum([r * w for r, w in zip(rewards, self._weights)])
        self.sendOrPrint(locals())
        return weighted_sum

    # buf = [collections.deque(maxlen=20) for _ in range(4)]
    def sendOrPrint(self, variables: dict):
        def get(key):
            return variables[key]

        # for s, b in zip(self.robot.getFootSlipVelocity(), self.buf):
        #     b.append(s)
        # print(np.array([np.mean(b) for b in self.buf]))
        # print(self.robot.getCostOfTransport())
        if any(strides := get('strides')):
            print(strides, SmallStridePenalty()(strides))
        # print(get('cot'), CostOfTransportReward()(get('cot')))
        # print(get('slips'))
        # print()
        # print(get('r_rate'), get('p_rate'), get('y_rate'))
        # print()
        # from burl.utils import udp_pub
        # data = {'hip_joints': tuple(self.robot.getJointPositions()[(0, 3, 6, 9),])}
        # udp_pub.send(data)
        pass

    def getRewardDetails(self):
        return self._details

    def reset(self):
        pass

    # def curriculumUpdate(self, episode_len):
    #     distance = np.dot(self.robot.position, self._cmd)
    #     return self._terrain.register(episode_len, distance)

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
