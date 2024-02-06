import numpy as np
from typing import Dict, Tuple
import networkx as nx

def obj_function(A0 : np.ndarray, A1: np.ndarray, W : np.ndarray) -> int:
    """Computes the objective function of our splex problem"""
    X = (A0 != A1)
    return (X * W).sum() // 2

# def obj_function(X: np.ndarray, W : np.ndarray):
#     return (X * W).sum()

def adjmatrix2adjmap(W: np.ndarray) -> Dict[Tuple[int, int], int]:
    """Converts and adjacency matrix into a adjacency list:
            (i,j) : w
    """
    return {(i,j): w for i,a in enumerate(W) for j,w in enumerate(a) if i<j} 


def is_splex( A: np.ndarray,  s: int) -> bool:
    """Checks if an adjacency matrix A is composed only by splexes"""
    G = nx.from_numpy_array(A)

    for cc in nx.connected_components(G):
        if not is_splex_component(cc, s, A):
            return False
        
    return True


def is_splex_component(component : set, s: int, A: np.ndarray) -> bool:
    """Checks if a subset of nodes is an splex in an adjacency matrix A"""
    component = list(component)
    neighbor_degrees = A[component].sum(axis=1)
    return (neighbor_degrees >= neighbor_degrees.shape[0] - s).sum() == len(component) 