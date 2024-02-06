import numpy as np
import numpy.typing as npt


import dataclasses as dc
from copy import deepcopy
from typing import Collection, Dict, List


@dc.dataclass
class Solution:
    # A: npt.NDArray[np.int_]
    W: npt.NDArray[np.int_]
    A1: npt.NDArray[np.int_]
    clusters: Dict[int, List[int]]
    _obj: int = -1

    @property
    def size(self):
        return self.W.shape[0]

    def copy(self):
        return Solution(self.W, self.A1.copy(), deepcopy(self.clusters))

    def obj(self) -> int:
        if self._obj != -1:
            return self._obj

        self._obj = (self.A1 * self.W).sum() - self.W[self.W<0].sum()
        self._obj = self._obj // 2
        return self._obj
    
    def plexes(self):
        clusters_ids = set()
        for plex in self.clusters.values():
            if id(plex) in clusters_ids:
                continue
            clusters_ids.add(id(plex))
            
            yield plex

    
    @staticmethod
    def build(A, W, A1, clusters):
        clusters = {k:s for s in [list(s) for s in clusters.values()] for k in s}
        W1 = W.copy()
        np.fill_diagonal(W1, W1.sum())
        W1[A==1] = - W1[A==1]

        return Solution(W1,A1,clusters)