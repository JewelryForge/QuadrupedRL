from abc import abstractmethod, ABCMeta
from typing import Union, Tuple

import numpy as np
import torch
from rtree import Rtree
from sklearn.linear_model import LinearRegression


class PolicyWrapper(object):
    def __init__(self, net, device):
        self.net = net
        self.device = torch.device(device)

    def __call__(self, obs):
        with torch.inference_mode():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            return self.net(obs).detach().cpu().numpy()


class GradIS(object, metaclass=ABCMeta):
    def __init__(
        self,
        min_key: Union[float, Tuple[float, ...]],
        max_key: Union[float, Tuple[float, ...]],
        max_len: int,
        grad_coef=1.0,
        max_weight=5.0,
    ):
        self._rtree = Rtree()
        self._min_key = min_key
        self._max_key = max_key
        if isinstance(self._min_key, float):
            self._interval = max_key - min_key
        else:
            self._interval = np.asarray(max_key) - np.asarray(min_key)
        self._max_len = max_len
        self._grad_coef = grad_coef
        self._max_weight = max_weight
        self._bbox = self._key_to_bbox(min_key, max_key)

        self._rtree.insert(-1, self._bbox)

        self._cur_len = 0
        self._total_len = 0
        self._buffer = []
        self._weights = []  # save grad weights
        self._weight_sum = 0.

    @abstractmethod
    def _key_to_bbox(self, key1, key2=None):
        raise NotImplementedError

    @abstractmethod
    def _bbox_to_key(self, bbox):
        raise NotImplementedError

    @property
    def samples(self):
        return iter(self)

    @property
    def particles(self):
        return zip(self._buffer, self._weights)

    def __iter__(self):
        for item in self._rtree.intersection(
            self._bbox, objects=True
        ):
            if item.id != -1:
                yield self._bbox_to_key(item.bbox), item.object

    def is_full(self):
        return self._cur_len == self._max_len

    def __len__(self):
        return self._cur_len

    def __repr__(self):
        return self._rtree.__repr__()

    def insert(self, key, value):
        self._rtree.insert(self._total_len, self._key_to_bbox(key), value)
        if self._cur_len < self._max_len:
            self._buffer.append(key)
            self._weights.append(1.0)
            self._weight_sum += 1.0
            self._cur_len += 1
        else:
            idx = self._total_len % self._max_len
            prev_key = self._buffer[idx]
            self._rtree.delete(
                self._total_len - self._max_len,
                self._key_to_bbox(prev_key)
            )
            self._buffer[idx] = key
            grad_weight = self.get_grad_weight(key)
            self._weight_sum += grad_weight - self._weights[idx]
            self._weights[idx] = grad_weight

        self._total_len += 1

    def get_grad_weight(self, key) -> float:
        neighbors = self.get_neighbors(key, self._interval / 10)
        if len(neighbors) < 5:
            return self._max_weight
        else:
            x, y = zip(*neighbors)
            x, y = np.array(x), np.array(y)
            reg = LinearRegression()
            reg.fit(x.reshape(-1, 1), y)
            return np.exp(abs(reg.coef_[0]) * self._grad_coef).clip(max=5.)

    def get_neighbors(self, key, radius):
        neighbors = []
        for item in self._rtree.intersection(
            self._key_to_bbox(key - radius, key + radius), objects=True
        ):
            if item.id != -1:
                neighbors.append((self._bbox_to_key(item.bbox), item.object))
        return neighbors

    def sample(
        self,
        random_gen: Union[np.random.Generator, np.random.RandomState],
        uniform_prob: float = 0.,
        normal_var: float = None,
    ):
        if not self.is_full() or random_gen.random() < uniform_prob:
            return random_gen.uniform(self._min_key, self._max_key)
        else:
            # importance sampling
            weights = np.array(self._weights) / self._weight_sum
            sample = random_gen.choice(self._buffer, p=weights)
            return np.clip(
                random_gen.normal(sample, normal_var),
                self._min_key, self._max_key
            ) if normal_var else sample


class GradIS1D(GradIS):
    def _key_to_bbox(self, key1, key2=None):
        if key2 is None:
            return key1, key1, key1, key1
        else:
            return key1, key1, key2, key2

    def _bbox_to_key(self, bbox):
        return bbox[0]
