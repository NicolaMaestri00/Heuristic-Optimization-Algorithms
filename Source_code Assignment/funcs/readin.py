import numpy as np
from typing import Tuple
import networkx as nx
import os

def readin(filename : str) -> Tuple[int, np.ndarray, np.ndarray]:
    with open(filename, 'r') as f:
        lines = f.readlines()
        nums = [list(map(int, line.split(' '))) for line in lines]
        s, node_num, edge_num, line_num = nums.pop(0)

        A = np.zeros((node_num, node_num), dtype=np.int8)
        W = np.zeros((node_num, node_num), dtype=np.int32)

        for i,j,a,w in nums:
            i -= 1
            j -= 1
            A[i,j] = a 
            A[j,i] = a 
            W[j,i] = w 
            W[i,j] = w 

    return s, A, W

import matplotlib.pyplot as plt
def show_adj_matrix(A: np.ndarray, W: np.ndarray = None):
    if W is not None:
        A = A*W
    G = nx.from_numpy_array(A)
    f = plt.figure(figsize=(3,3))
    nx.draw(G, with_labels=True)
    plt.show()

def writeout(A, A1, folder, prob_name, addendum=''):
    if not os.path.exists('output/'+folder):
        os.makedirs('output/'+folder)
    ii, jj = np.where(A != A1)
    ii+=1
    jj+=1
    with open('output/'+folder+prob_name+addendum+'.txt', '+w') as f:
        f.write(prob_name + '\n')
        for i,j in zip(ii, jj):
            if i>=j:
                continue
            f.write(f'{i} {j}\n')


# s, A, W = readin('data/test1.txt')
# print(A)
# show_adj_matrix(A)
