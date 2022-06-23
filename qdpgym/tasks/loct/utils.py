import collections
from typing import Union, Tuple, List, Any, Iterable, Callable

import numpy as np
import rtree
import torch
from rtree import Rtree
from sklearn.linear_model import LinearRegression

from qdpgym.utils import tf, Natural


class PolicyWrapper(object):
    def __init__(self, net, device):
        self.net = net
        self.device = torch.device(device)

    def __call__(self, obs):
        with torch.inference_mode():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            return self.net(obs).detach().cpu().numpy()


class GradISNaive(object):
    """
    Adaptively change the sample weight by the gradient of reward.
    May lead to significant sample impoverishment.
    """

    def __init__(
        self,
        dim: int,
        min_key: Union[float, Tuple[float, ...]],
        max_key: Union[float, Tuple[float, ...]],
        max_len: int,
        grad_coef=1.0,
        max_weight=5.0,
        neighborhood_ratio=0.1,
    ):
        self._rtree = Rtree()
        self._min_key = min_key
        self._max_key = max_key
        if isinstance(self._min_key, float):
            self._interval = max_key - min_key
        else:
            self._interval = np.asarray(max_key) - np.asarray(min_key)
        self._max_len = max_len
        self._dim = dim
        self._grad_coef = grad_coef
        self._max_weight = max_weight
        self._nbh_size = neighborhood_ratio * self._interval
        self._bbox = self._key_to_bbox(min_key, max_key)

        self._rtree.insert(-1, self._bbox)

        self._cur_len = 0
        self._total_len = 0
        self._buffer = []
        self._weights = []  # save grad weights
        self._weight_sum = 0.

    def _key_to_bbox(self, key1, key2=None):
        if key2 is None:
            key2 = key1
        if self._dim == 1:
            return key1, key1, key2, key2
        elif self._dim == 2:
            return (*key1, *key2)
        else:
            raise NotImplementedError

    def _bbox_to_key(self, bbox):
        if self._dim == 1:
            return bbox[0]
        elif self._dim == 2:
            return bbox[:2]
        else:
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
        neighbors = self.get_neighbors(key, self._nbh_size)
        if len(neighbors) < 5:
            return self._max_weight
        else:
            x, y = zip(*neighbors)
            x, y = np.array(x), np.array(y)
            reg = LinearRegression()
            reg.fit(x.reshape(-1, 1), y)
            return np.exp(
                abs(reg.coef_[0]) * self._grad_coef
            ).clip(max=self._max_weight)

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


class GradIS(object):
    def __init__(
        self,
        dim: int,
        min_key: Union[float, Tuple[float, ...]],
        max_key: Union[float, Tuple[float, ...]],
        approx_max_len: int,
        grad_coef=1.0,
        max_weight=5.0,
        neighborhood_ratio=0.1,
    ):
        self._rtree = Rtree()
        self._dim = dim
        self._min_key = min_key
        self._max_key = max_key
        if isinstance(self._min_key, float):
            self._interval = max_key - min_key
        else:
            self._interval = np.asarray(max_key) - np.asarray(min_key)
        self._approx_max_len = approx_max_len
        self._max_density = self._approx_max_len * neighborhood_ratio ** dim
        self._grad_coef = grad_coef
        self._max_weight = max_weight
        self._nbh_size = neighborhood_ratio * self._interval
        self._bbox = self._key_to_bbox(min_key, max_key)

        self._rtree.insert(-1, self._bbox)

        self._cur_len = 0
        self._total_len = 0
        self._buffer_weights = collections.defaultdict(float)
        self._weight_sum = 0.

    def _key_to_bbox(self, key1, key2=None):
        if key2 is None:
            key2 = key1
        if self._dim == 1:
            return key1, key1, key2, key2
        elif self._dim == 2:
            return (*key1, *key2)
        else:
            raise NotImplementedError

    def _bbox_to_key(self, bbox):
        if self._dim == 1:
            return bbox[0]
        elif self._dim == 2:
            return bbox[:2]
        else:
            raise NotImplementedError

    def _key_dist(self, key1, key2):
        if self._dim == 1:
            return abs(key1 - key2)
        elif self._dim == 2:
            return tf.vnorm(np.asarray(key1) - np.asarray(key2))
        else:
            raise NotImplementedError

    @property
    def samples(self) -> Iterable[Tuple[Any, float]]:
        return iter(self)

    @property
    def particles(self):
        return self._buffer_weights.items()

    @property
    def initialized(self):
        return self._total_len >= self._approx_max_len

    def __iter__(self):
        for item in self._rtree.intersection(
            self._bbox, objects=True
        ):
            if item.id != -1:
                yield self._bbox_to_key(item.bbox), item.object

    def __len__(self):
        return self._cur_len

    def __repr__(self):
        return self._rtree.__repr__()

    def insert(self, key, value):
        raw_neighbors = self.get_neighbors(key, self._nbh_size, raw=True)

        if key in self._buffer_weights:
            # remove duplicated key
            dup_idx = None
            for idx, item in enumerate(raw_neighbors):
                item_key = self._bbox_to_key(item.bbox)
                if np.array_equal(item_key, key):
                    # print('duplicated key', key)
                    self._weight_sum -= self._buffer_weights[key]
                    self._rtree.delete(item.id, item.bbox)
                    dup_idx = idx
                    break
            if dup_idx is not None:
                raw_neighbors.pop(dup_idx)
        elif len(raw_neighbors) > self._max_density:
            # delete the obsolete particle
            # obsolete: rtree.index.Item = min(raw_neighbors, key=lambda x: x.id)
            # or delete the nearest particle?
            obsolete: rtree.index.Item = min(
                raw_neighbors, key=lambda x: self._key_dist(self._bbox_to_key(x.bbox), key)
            )
            raw_neighbors.remove(obsolete)
            self._rtree.delete(obsolete.id, obsolete.bbox)
            self._weight_sum -= self._buffer_weights.pop(self._bbox_to_key(obsolete.bbox))
        else:
            self._cur_len += 1

        self._total_len += 1
        self._rtree.insert(self._total_len, self._key_to_bbox(key), value)
        # calculate weight from gradient
        neighbors = [(self._bbox_to_key(item.bbox), item.object) for item in raw_neighbors]
        neighbors.append((key, value))
        grad_weight = self.get_weight_from_grad(key, neighbors)
        self._weight_sum += grad_weight
        self._buffer_weights[key] = grad_weight

    def get_weight_from_grad(self, key, neighbors=None) -> float:
        if neighbors is None:
            neighbors = self.get_neighbors(key, self._nbh_size)
        if len(neighbors) < 5:
            return 1.0
        else:
            x, y = zip(*neighbors)
            x, y = np.array(x), np.array(y)
            reg = LinearRegression()
            # TODO: dim >= 2
            reg.fit(x.reshape(-1, 1), y)
            r = reg.predict(np.reshape(key, (1, 1))).clip(0., 1.).item()
            return np.clip(
                np.exp(abs(reg.coef_[0]) * self._grad_coef) +
                np.exp(-((r - 0.7) / 0.7) ** 2),
                0., self._max_weight
            ).item()

            # grad = 2 * r * math.sqrt(-math.log(r))
            # return np.clip(abs(grad), 0., self._max_weight).item()
            # # return np.exp(
            # #     abs(grad) * self._grad_coef
            # # ).clip(max=self._max_weight)

    def get_neighbors(
        self, key, radius, raw: bool = False
    ) -> Union[List[rtree.index.Item], List[Tuple[Any, float]]]:
        neighbors = []
        for item in self._rtree.intersection(
            self._key_to_bbox(key - radius, key + radius), objects=True
        ):
            if item.id != -1:
                neighbors.append(
                    item if raw else (
                        self._bbox_to_key(item.bbox), item.object
                    ))
        return neighbors

    def sample(
        self,
        random_gen: Union[np.random.Generator, np.random.RandomState],
        uniform_prob: float = 0.,
        normal_var: float = None,
    ):
        if not self.initialized or random_gen.random() < uniform_prob:
            return random_gen.uniform(self._min_key, self._max_key)
        else:
            # importance sampling
            # TODO: a dict for saving index and two lists
            #   for saving samples and weights may be better?
            buffer, weights = zip(*self._buffer_weights.items())
            weights = np.array(weights) / self._weight_sum
            sample = random_gen.choice(buffer, p=weights)
            return np.clip(
                random_gen.normal(sample, normal_var),
                self._min_key, self._max_key
            ) if normal_var else sample


class AlpIS(object):
    def __init__(
        self,
        dim: int,
        min_key: Union[float, Tuple[float, ...], np.ndarray],
        max_key: Union[float, Tuple[float, ...], np.ndarray],
        window_size: int,
        default_distribution: Callable[[Union[np.random.Generator, np.random.RandomState]], Any] = None,
        neighborhood_ratio=0.1,
        min_lp=1e-4,
    ):
        self._dim = dim
        self._min_key = min_key
        self._max_key = max_key
        self._interval = np.array(max_key) - np.array(min_key)
        self._nbh_size = neighborhood_ratio * self._interval
        self._window_size = window_size
        self._max_density = self._window_size * neighborhood_ratio ** dim
        self._min_lp = min_lp
        if default_distribution is None:
            self._explore = lambda r: r.uniform(self._min_key, self._max_key)
        else:
            self._explore = default_distribution

        self._buffer = Rtree()
        self._history = Rtree()
        self._bbox = self._key_to_bbox(min_key, max_key)
        self._history_idx = Natural()
        self._window = collections.deque(maxlen=window_size)

        self._total_len = 0
        self._samples = []
        self._weights = []
        self._init = False

    def _key_to_bbox(self, key1, key2=None):
        if key2 is None:
            key2 = key1
        if self._dim == 1:
            return key1, key1, key2, key2
        if self._dim == 2:
            return (*key1, *key2)
        raise NotImplementedError

    def _bbox_to_key(self, bbox):
        if self._dim == 1:
            return bbox[0]
        if self._dim == 2:
            return bbox[:2]
        raise NotImplementedError

    def _key_dist(self, key1, key2):
        if self._dim == 1:
            return abs(key1 - key2)
        if self._dim == 2:
            return tf.vnorm(np.asarray(key1) - np.asarray(key2))
        raise NotImplementedError

    @property
    def progresses(self) -> Iterable[Tuple[Any, float]]:
        for item in self._history.intersection(self._bbox, objects=True):
            if item.id != -1:
                yield self._bbox_to_key(item.bbox), item.object

    @property
    def particles(self):
        return zip(self._samples, self._weights)

    @property
    def is_init(self):
        return self._init

    def insert(self, key, value):
        self._total_len += 1
        self._buffer.insert(self._total_len, self._key_to_bbox(key), value)
        self._window.append((key, value))

        if self._total_len % self._window_size == 0:
            self._samples.clear()
            self._weights.clear()
            if self._init or len(self._history):
                self._init = True
                for k, _ in self.progresses:
                    self._samples.append(k)
                    self._weights.append(self._compute_alp(k))
                weight_sum = np.sum(self._weights)
                if weight_sum != 0.:
                    self._weights = (np.array(self._weights) / weight_sum).tolist()

            self._patch_density_merge(self._window)
            self._buffer = Rtree()

    def sample(
        self,
        random_gen: Union[np.random.Generator, np.random.RandomState],
        uniform_prob: float = 0.,
        normal_var: float = None,
    ):
        if not self._init or random_gen.random() < uniform_prob:
            return self._explore(random_gen)
        else:
            # importance sampling
            sample = random_gen.choice(self._samples, p=self._weights)
            return np.clip(
                random_gen.normal(sample, normal_var),
                self._min_key, self._max_key
            ) if normal_var else sample

    def _patch_density_merge(self, window):
        for k, v in window:
            neighbors = self._get_neighbors(self._history, k)
            if len(neighbors) > self._max_density:
                # delete the nearest particle
                obsolete: rtree.index.Item = min(
                    neighbors, key=lambda x: self._key_dist(self._bbox_to_key(x.bbox), k)
                )
                self._history.delete(obsolete.id, obsolete.bbox)
            self._history.insert(next(self._history_idx), self._key_to_bbox(k), v)

    def _compute_alp(self, key) -> float:
        n_history = self._get_neighbor_objects(self._history, key)
        n_buffer = self._get_neighbor_objects(self._buffer, key)
        # TODO: Remove Duplicated items
        if len(n_history) > 3 and len(n_buffer) > 3:
            v_history = np.array(n_history).mean()
            v_buffer = np.array(n_buffer).mean()
            alp = max(abs(v_buffer - v_history), self._min_lp)
        else:
            alp = self._min_lp
        return alp

    def _get_neighbors(self, rtree_obj, key) -> List[rtree.index.Item]:
        neighbors = []
        bbox = self._key_to_bbox(key - self._nbh_size, key + self._nbh_size)
        for item in rtree_obj.intersection(bbox, objects=True):
            if item.id != -1:
                neighbors.append(item)
        return neighbors

    def _get_neighbor_objects(self, rtree_obj, key) -> List[float]:
        neighbors = []
        bbox = self._key_to_bbox(key - self._nbh_size, key + self._nbh_size)
        for item in rtree_obj.intersection(bbox, objects=True):
            if item.id != -1:
                neighbors.append(item.object)
        return neighbors


class GradIS1D(GradIS):
    def __init__(
        self,
        min_key: Union[float, Tuple[float, ...]],
        max_key: Union[float, Tuple[float, ...]],
        max_len: int,
        grad_coef=1.0,
        max_weight=5.0,
        neighborhood_ratio=0.1,
    ):
        super().__init__(
            1,
            min_key,
            max_key,
            max_len,
            grad_coef,
            max_weight,
            neighborhood_ratio,
        )
