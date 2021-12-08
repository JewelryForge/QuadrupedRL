from burl.sim import Terrain, RandomUniformTerrain, PlainTerrain, SlopeTerrain
from burl.utils import g_cfg, logger


class BasicTerrainManager(object):
    def __init__(self):
        self.terrain: Terrain

    def register(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        return getattr(self.terrain, item)

    def reset(self):
        pass


def makeStandardRoughTerrain(bullet_client, roughness=None, seed=None):
    if roughness is None:
        roughness = g_cfg.trn_roughness
    return RandomUniformTerrain(
        bullet_client, size=g_cfg.trn_size, downsample=g_cfg.trn_downsample,
        roughness=roughness, resolution=g_cfg.trn_resolution, offset=g_cfg.trn_offset, seed=seed)


class SlopeTerrainManager(BasicTerrainManager):
    def __init__(self, bullet_client):
        super().__init__()
        self.terrain = SlopeTerrain(bullet_client, size=g_cfg.trn_size, slope=g_cfg.trn_slope, resolution=0.1)

    def reset(self):
        self.terrain = SlopeTerrain(self.bullet_client, size=g_cfg.trn_size, slope=g_cfg.trn_slope, resolution=0.1)


class PlainTerrainManager(BasicTerrainManager):
    def __init__(self, bullet_client):
        super().__init__()
        self.terrain = PlainTerrain(bullet_client)

    def reset(self):
        self.terrain = PlainTerrain(self.bullet_client)


class FixedRoughTerrainManager(BasicTerrainManager):
    def __init__(self, bullet_client, seed=None):
        super().__init__()
        self.seed = seed
        self.terrain = makeStandardRoughTerrain(bullet_client, seed=seed)

    def reset(self):
        self.terrain = makeStandardRoughTerrain(self.bullet_client, self.seed)


class TerrainCurriculum(BasicTerrainManager):
    def __init__(self, bullet_client):
        super().__init__()
        self.bullet_client = bullet_client
        self.terrain = makeStandardRoughTerrain(self.bullet_client, 0.0)
        self.counter = 0
        self.difficulty = 0.0
        self.difficulty_level = 0
        self.combo, self.miss = 0, 0

    def decreaseLevel(self):
        if self.difficulty_level > 0:
            self.difficulty -= g_cfg.difficulty_step
            self.difficulty_level -= 1
            logger.debug(f'decrease level, current {self.difficulty_level}')

    def increaseLevel(self):
        if self.difficulty < g_cfg.max_difficulty:
            self.difficulty += g_cfg.difficulty_step
            self.difficulty_level += 1
            logger.debug(f'increase level, current {self.difficulty_level}')

    def register(self, episode_len, distance):  # FIXME: THIS DISTANCE IS ON CMD DIRECTION
        self.counter += 1
        if episode_len == g_cfg.max_sim_iterations:
            self.miss = 0
            self.combo += 1
        else:
            self.combo = 0
            self.miss += 1
        logger.debug(f'Miss{self.miss} Combo{self.combo} distance{distance:.2f}')
        if self.miss and self.miss % g_cfg.miss_threshold == 0:
            self.decreaseLevel()
            return True
        elif self.combo and self.combo % g_cfg.combo_threshold == 0:
            lower, upper = g_cfg.distance_threshold
            if distance > upper:
                self.increaseLevel()
                return True
            # elif distance < lower:
            #     self.decreaseLevel()

        return False

    def reset(self):
        self.terrain = makeStandardRoughTerrain(self.bullet_client, self.difficulty)
