import numpy as np
import random
from .Solution import Solution
from .Greedy import weighted_karger
from .splex_utils import is_splex
from tqdm import tqdm

class LargeNeighborhoodSearch:
    def __init__(self, destroyers, repairers, s, gamma = 0.1, epochs = 1000, temperature = 1e3, stop_temp = 50 , alpha = 0.9) -> None:
        self.s = s
        self.gamma = gamma
        self.destroyers = destroyers
        self.repairers  = repairers

        self.len_dest = len(destroyers)
        self.len_repa = len(repairers)

        self.rho_neg = np.ones((len(destroyers), ), dtype=float)
        self.rho_pos = np.ones((len(repairers), ), dtype=float)
        self.dest_succ = np.zeros((len(destroyers), ))
        self.dest_att  = np.zeros((len(destroyers), ), dtype=int)
        self.repa_succ  = np.zeros((len(repairers), ))
        self.repa_att  = np.zeros((len(repairers), ), dtype=int)

        self.epochs = epochs
        self.periods = np.ceil(np.log(stop_temp/temperature) / np.log(alpha)).astype(int)

        self.temperature = temperature
        self.alpha = alpha

        self.best = None
        self.best_score = None
        

    def search(self, x0:Solution) -> Solution:
        self.best = x0
        self.best_score = x0.obj()

        for j in range(self.periods):
            self.rho_neg /= self.rho_neg.sum()
            self.rho_pos /= self.rho_pos.sum()
            for i in range(self.epochs):
                x0 = self.step(x0)

            self.rho_neg = (1-self.gamma) *self.rho_neg + self.gamma * self.dest_succ/self.dest_att
            self.rho_pos = (1-self.gamma) *self.rho_pos + self.gamma * self.repa_succ/self.repa_att
            self.temperature = self.alpha * self.temperature
            # print(f'END EPOCH {j}\t Current best {self.best_score}\t Temperature {self.temperature}')
            # print(self.rho_neg, self.rho_pos)
        return self.best

    def step(self, x0: Solution) -> Solution:
        dest = np.random.choice(len(self.destroyers), p=self.rho_neg)
        repa = np.random.choice(self.len_repa, p=self.rho_pos)

        x1, clusters = self.destroyers[dest].destroy(x0)
        for cl in clusters:
            cluster_idx = np.ix_(cl, cl)
            A1 = self.repairers[repa](x0.W, plex=cl, s=self.s)
            x1.A1[cluster_idx] = A1[cluster_idx]

        self.repa_att[repa] += 1
        self.dest_att[dest] += 1
        # assert is_splex(x1.A1, self.s), f'Is not anymore S Plex. Destroyer={dest}, Repairer={repa}'
        # assert all(np.unique(cl).shape[0] == len(cl) for cl in x1.clusters.values())

        p = np.exp((x0.obj() - x1.obj())/self.temperature)

        if p >= 1 or p > np.random.random():
            x0 = x1
            self.repa_succ[repa] += min(1,p)
            self.dest_succ[dest] += min(1,p)

            if x1.obj() < self.best_score:
                self.best_score = x1.obj()
                self.best = x1
        

        return x0



        

        

