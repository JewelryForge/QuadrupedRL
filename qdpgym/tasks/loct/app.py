import qdpgym
from qdpgym import sim
from qdpgym.tasks import loct

__all__ = ['LocomotionApp']


class LocomotionApp(sim.Application):
    def __init__(self, policy):
        robot = sim.Aliengo(500, 'actuator_net', noisy=False)
        task = loct.LocomotionV0()
        if qdpgym.sim_engine == qdpgym.Sim.BULLET:
            arena = sim.TerrainBase()
            task.add_hook(sim.ExtraViewerHook())
            task.add_hook(sim.RandomTerrainHook())
            task.add_hook(sim.RandomPerturbHook())
            task.add_hook(loct.GamepadCommanderHook())
            # task.add_hook(sim.StatisticsHook())
            # task.add_hook(hooks.VideoRecorderBtHook())
        else:
            raise NotImplementedError
        env = sim.QuadrupedEnv(robot, arena, task)
        # task.add_reward('UnifiedLinearReward')
        super().__init__(robot, env, task, policy)
