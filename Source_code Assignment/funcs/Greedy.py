import numpy as np
from funcs import adjmatrix2adjmap
from itertools import chain
import random
import numba as nb
import numba.types as nbt
import numpy.typing as npt
from typing import List

class GreedySPlex:
    TERROR_VALUE = 99999
    def __init__(self, adjacency_matrix : np.ndarray, weight_matrix: np.ndarray, s: int):
        self.A = adjacency_matrix
        self.W = weight_matrix.copy()
        np.fill_diagonal(self.W, GreedySPlex.TERROR_VALUE)
        self.W[adjacency_matrix==1] *= -1
        self.s = s 
        self.weightmap = adjmatrix2adjmap(self.W)

    def random_solution(self):
        # We start with the graph with all edges removed
        # and we try to add them back one at a time
        edges = [e for e in self.weightmap.keys() if self.weightmap[e]<0]
        random.shuffle(edges)
        return self.solve(edges)

    def solution(self):
        # We start with the graph with all edges removed
        # and we try to add them back one at a time (sorted by weight)
        edges = [e for e in self.weightmap.keys() if self.weightmap[e]<0]
        edges_by_weight = sorted(edges, key=self.weightmap.get)
        return self.solve(edges_by_weight)

    def solve(self, edges: dict):
        splexes = {i:{i} for i in range(self.A.shape[0])}
        A1 = np.zeros(self.A.shape, dtype=int)

        for i,j in edges:
            # print(f"Considering [{i},{j}]")
            s_i = splexes[i]
            s_j = splexes[j]
            plex_size = len(s_i) + len(s_j)
            
            if s_i is s_j or plex_size <= (self.s+1):
                # print('\tFast merge.')
                # if they are less than S+1 they are always a legit s-plex
                self.merge(i,j,splexes, A1)
                continue

            cost = 0
            added_edges = dict()
            candidate_splex = list(chain(s_i, s_j))

            # We now try to see if merging the 2 splexes is convenient or not
            # For each of the nodes we check how many edges we need to add to make it a valid splex
            # we then add the "less expensive" edges
            # if the operation is convinient we perform it, otherwise we abort it

            W1 = self.W.copy()
            # in order to constrain our choice of edges to only those in the considered splexes,
            # We're going to use a Terror_Value (very high number) to make the algorithm pick only the desired edges

            # set the weights of candidate plex to very low value in order to make the algorithm choose that
            W1[np.ix_(candidate_splex, candidate_splex)] -= GreedySPlex.TERROR_VALUE
            # W1[list(s_j), list(s_i)] -= GreedySPlex.TERROR_VALUE
            # set the weights of already present edges to terror value
            W1[A1 == 1] = GreedySPlex.TERROR_VALUE
            np.fill_diagonal(W1, GreedySPlex.TERROR_VALUE)
            # print(f'\tCandidate Merging {s_i} U {s_j}')
            missing_edges = {k:plex_size - self.s - A1[k].sum() for k in candidate_splex}
            for k in candidate_splex:
                m = missing_edges[k]
                # print(f'\t\tConsidering {k}, missing: {m}')
                if m <= 0:
                    added_edges[k] = []
                    continue

                weights = W1[k]
                nodes = np.argpartition(weights, m)[:m]
                # print(f'\t\t{weights}')
                # print(f'\t\t{nodes}')

                added_edges[k] = nodes
                for node in nodes:
                    missing_edges[node] -= 1
                cost += self.W[k,nodes].sum()
                W1[k,nodes] = GreedySPlex.TERROR_VALUE
                W1[nodes,k] = GreedySPlex.TERROR_VALUE

            # if the computed cost indicates that there's a gain
            # print(f'\tCost: {cost}')
            if cost < 0:
                # print('\tMerge.')
                self.merge(i,j,splexes, A1, added_edges)

        return A1, splexes
    
    def merge(self, i: int, j: int, splexes: dict, A1: np.ndarray, added_edges : dict = None):
        # Set edge
        A1[i,j] = 1
        A1.T[i,j] = 1
        # Merge splexes
        if splexes[i] is not splexes[j]:
            splexes[i] = splexes[i] | splexes[j] # union
            for k in splexes[i]:
                splexes[k] = splexes[i]

                if added_edges is not None:
                    A1[k, added_edges[k]] = 1
                    A1[added_edges[k], k] = 1

###


@nb.njit
def delete(arr, idx):
    '''deletes a column,row from a matrix'''
    mask = np.ones(arr.shape[0], dtype=np.bool_)
    mask[idx] = False
    return arr[mask][:,mask]

@nb.njit
def rand_choice(odds):
    """
    :param odds: A 1D numpy array of values to sample from.
    :return: A random sample from the given array with a given probability.
    """
    freq = np.exp(odds) # softmax
    # freq = odds - odds.min()

    prob = freq/freq.sum()
    return np.searchsorted(np.cumsum(prob), np.random.random(), side="right")

@nb.njit(nbt.UniTuple(nb.int32,2)(nb.int32[:,:]))
def random_idx(A : npt.NDArray[np.int32]):
    '''Extracts a random edge out of an adjacency matrix'''
    idx = np.triu_indices(A.shape[0], k=1)
    p = np.zeros_like(idx[0], dtype=np.int32)
    for k, (i,j) in enumerate(zip(*idx)):
        p[k] = -A[i,j]

    e = rand_choice(p)
    i,j = idx[0][e], idx[1][e]

    return i,j


@nb.njit#(nbt.UniTuple(nbt.List(nbt.List(nb.int32)), )(nb.int32[:,:]))
def weighted_karger(W : npt.NDArray[np.int32], random=False):
    '''Inspired by Karger algroithm for probabilistic min-cut. 
    It searches the max-cut over the weighted edges and partitions the graph into clusters
    '''
    A = W.copy()
    cluster = [[i] for i in range(W.shape[0])]
    nodes = list(range(W.shape[0]))

    while A.shape[0] > 2 and A.min() < 0 :
        if random:
            i,j = random_idx(A)
        else: 
            i,j =  divmod(A.argmin(), A.shape[1]) # deterministic

        cluster[i] += cluster[j]

        A[i] += A[j]
        A[:,i] += A[:,j]
        A = delete(A, j)

        del cluster[j]
        del nodes[j]


    return cluster


def deletion_heuristic(A : npt.NDArray[np.int32], plex: List[int], s:int):
    '''given some plex nodes remove all the possible edges starting from the biggest one until possible'''
    if len(plex) == 1:
        return np.zeros_like(A, dtype=np.int32)
    cluster_idx = np.ix_(plex, plex)
    Ac = A[cluster_idx]     # Adjacency matrix for the cluster only
    min_edges = len(plex) - s
    sorted_edges = np.argsort(Ac, axis=-1)[:,s::-1]
    A1 = np.ones_like(Ac, dtype=np.int32)
    np.fill_diagonal(A1, 0)
    
    for i in range(Ac.shape[0]):
        extra_edges = min(A1[i].sum() - min_edges, A1.shape[1])
        for k in range(extra_edges):
            to_remove = sorted_edges[i,k]
            if Ac[i,to_remove] < 0:
                break

            if A1[i,to_remove] == 0 or A1[to_remove].sum() <= min_edges:
                continue
            A1[i,to_remove] = 0
            A1[to_remove,i] = 0


    res = np.zeros_like(A, dtype=np.int32)
    res[cluster_idx] = A1
    return res

def insertion_heuristic(A : npt.NDArray[np.int32], plex: List[int], s:int):
    '''given some plex nodes remove all the possible edges starting from the biggest one until possible'''
    if len(plex) == 1:
        return np.zeros_like(A, dtype=np.int32)
    cluster_idx = np.ix_(plex, plex)
    Ac = A[cluster_idx]     # Adjacency matrix for the cluster only
    min_edges = len(plex) - s
    A1 = np.zeros_like(Ac, dtype=np.int32)
    A1[Ac < 0] = 1
    Ac[Ac < 0] = Ac[0,0]

    for i in range(Ac.shape[0]):
        missing_edges = min_edges - A1[i].sum()
        if missing_edges <= 0:
            continue
        edges = np.argpartition(Ac[i], missing_edges)[:missing_edges]
        A1[i,edges] = 1
        A1[edges,i] = 1
        Ac[i, edges] = 99999
        Ac[edges, i] = 99999
    
    res = np.zeros_like(A, dtype=np.int32)
    res[cluster_idx] = A1
    return res
        


class Karger:
    def __init__(self, adjacency_matrix : np.ndarray, weight_matrix: np.ndarray, s: int):
        self.A = adjacency_matrix
        self.W = weight_matrix.copy()
        np.fill_diagonal(self.W, self.W.sum())
        self.W[adjacency_matrix==1] *= -1
        self.s = s 


    def random_solution(self):
        clusters = weighted_karger(self.W, random=True)
        A1 = sum(deletion_heuristic(self.W, plex=cl, s=self.s) for cl in clusters)

        splexes = {i:plex for plex in clusters for i in plex}
        return A1, splexes
    
    def solution(self):
        clusters = weighted_karger(self.W, random=False)
        A1 = sum(deletion_heuristic(self.W, plex=cl, s=self.s) for cl in clusters)

        splexes = {i:plex for plex in clusters for i in plex}
        return A1, splexes



