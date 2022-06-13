import abc
from typing import Union, Tuple

import numpy as np
from dm_control import composer
from scipy.interpolate import interp2d

from qdpgym.sim.abc import NUMERIC, Terrain


def _process_size(size):
    if isinstance(size, (int, float)):
        return size, size
    else:
        return tuple(size)


class Arena(composer.Arena, Terrain, metaclass=abc.ABCMeta):
    pass


class TexturedTerrainBase(Arena):
    reflectance = 0.2

    def _build(self, size, name=None):
        super()._build(name=name)
        self._x_size, self._y_size = _process_size(size)

        self._mjcf_root.visual.headlight.set_attributes(
            ambient=[.4, .4, .4], diffuse=[.8, .8, .8], specular=[.1, .1, .1])

        self._ground_texture = self._mjcf_root.asset.add(
            'texture', name='groundplane', rgb1=[.2, .3, .4], rgb2=[.1, .2, .3],
            type='2d', builtin='checker', width=200, height=200,
            mark='edge', markrgb=[0.8, 0.8, 0.8])
        # Makes white squares exactly 1x1 length units.
        self._ground_material = self._mjcf_root.asset.add(
            'material', name='groundplane', texrepeat=[2, 2],
            texuniform=True, reflectance=self.reflectance, texture=self._ground_texture)

        self._terrain_geom = None

    @property
    def ground_geoms(self):
        return self._terrain_geom,

    def regenerate(self, random_state):
        pass

    @property
    def size(self):
        return self._x_size, self._y_size

    def out_of_range(self, x, y):
        return abs(x) > self._x_size / 2 - 1 or abs(y) > self._y_size / 2 - 1


class Plain(TexturedTerrainBase):
    def _build(self, size, name='groundplane'):
        super()._build(name=name, size=size)

        # Build groundplane.
        self._terrain_geom = self._mjcf_root.worldbody.add(
            'geom', type='plane', name='terrain', material=self._ground_material,
            size=[self._x_size / 2, self._x_size / 2, 0.25])
        self._terrain_geom.friction = [1, 0.005, 0.0001]

    def get_height(self, x, y):
        return 0.0

    def get_normal(self, x, y):
        return np.array((0., 0., 1.))


class Hills(TexturedTerrainBase):
    def _build(self, size, resol: float,
               *rough_dsp: Tuple[NUMERIC, NUMERIC],
               name='hills'):
        super()._build(name=name, size=size)
        self._resol = resol
        self._ncol = int(self._x_size / resol) + 1
        self._nrow = int(self._y_size / resol) + 1
        self._rough_dsp = rough_dsp

        self._mjcf_root.asset.add(
            'hfield', name=f'{name}_hfield',
            nrow=self._nrow, ncol=self._ncol,
            size=[self._x_size / 2, self._y_size / 2, 1, 0.1])
        self._terrain_geom = self._mjcf_root.worldbody.add(
            'geom', name='terrain', type='hfield', material=self._ground_material,
            pos=(0, 0, -0.01), hfield=f'{name}_hfield')

    def initialize_episode(self, physics, random_state: np.random.RandomState):
        physics.model.hfield_data[...] = self.regenerate(random_state)

    def regenerate(self, random_state):
        hfield_data = np.zeros((self._nrow, self._ncol))
        for rough, dsp in self._rough_dsp:
            x = np.arange(-3, int(self._ncol / dsp) + 4)
            y = np.arange(-3, int(self._nrow / dsp) + 4)
            hfield_dsp = random_state.uniform(0, rough, (x.size, y.size))
            terrain_func = interp2d(x, y, hfield_dsp, kind='cubic')
            x_usp = np.arange(self._ncol) / dsp
            y_usp = np.arange(self._nrow) / dsp
            hfield_data += terrain_func(x_usp, y_usp)
        return hfield_data.reshape(-1)
