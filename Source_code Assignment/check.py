from funcs import is_splex, readin
import numpy as np

with open('output/data/competition_instances/heur051_n_300_m_20122.txt', 'r') as f:
    node_num = int(f.readline().split('_')[2])
    lines = f.readlines()
    nums = [list(map(int, line.split(' '))) for line in lines]
    # s, node_num, edge_num, line_num = nums.pop(0)

    A1 = np.zeros((node_num, node_num), dtype=np.int8)

    for i,j in nums:
        i -= 1
        j -= 1
        A1[i,j] = 1 
        A1[j,i] = 1 

S,A,W = readin('data/competition_instances/heur051_n_300_m_20122.txt')


A[A1 == 1] = 1 - A[A1 == 1]
print(is_splex(A,S))

