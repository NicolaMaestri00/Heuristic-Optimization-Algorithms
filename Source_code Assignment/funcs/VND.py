import numpy as np

from .Solution import Solution
from .neighborhood import Neighborhood
from typing import Collection
import random

class partial_local_search:
    def __init__(self, neighborhoods : Collection[Neighborhood], len_loc_search: int=10) -> None:
        self.neighborhoods = neighborhoods
        self.len_loc_search = len_loc_search
        self.history = []

    def search(self, x0 : Solution) -> Solution:
        x = x0
        for i in range(self.len_loc_search):
            for neighborhood in self.neighborhoods:
                # First Improvement
                for x1 in neighborhood.neighbors(x):
                    if x1.obj() < x.obj():
                        x = x1
                        self.history.append(x1.obj())
                        break
                else:
                    break

        return x

class VariableNeighborhoodDescent:
    def __init__(self, neighborhoods : Collection[Neighborhood], mode = 'first') -> None:
        assert mode in ['first', 'best', 'random']
        self.neighborhoods = neighborhoods
        self.mode = mode

        self.history = []

    def search(self, x0 : Solution) -> Solution:
        x = x0
        improved = True
        while improved:
            improved = False
            for neighborhood in self.neighborhoods:
                if self.mode == 'best':
                    xs = neighborhood.neighbor_list(x)
                    if len(xs) == 0:
                        continue
                    x1 = min(xs, key = lambda x: x.obj())

                    if x1.obj() < x.obj():
                        print(f'Found {len(self.history)}th improvement at {neighborhood}. Delta={x.obj()-x1.obj()}')
                        x = x1
                        improved = True
                        self.history.append(x1.obj())
                        break
                elif self.mode == 'first':
                    # First Improvement
                    for x1 in neighborhood.neighbors(x):
                        if x1.obj() < x.obj():
                            # print(f'Found {len(self.history)}th improvement at {neighborhood}. Delta={x.obj()-x1.obj()}')
                            x = x1
                            improved = True
                            self.history.append(x1.obj())
                            break
                    else:
                        break
                elif self.mode == 'random':
                    for _ in range(1000):
                        x1 = neighborhood.shaking(x)
                        if x1.obj() < x.obj():
                            # print(f'Found {len(self.history)}th improvement at {neighborhood}. Delta={x.obj()-x1.obj()}')
                            x = x1
                            improved = True
                            self.history.append(x1.obj())
                            break
                    else:
                        break

                    

        return x
                