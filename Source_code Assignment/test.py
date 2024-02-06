import numpy as np
from funcs import writeout, readin, insertion_heuristic, deletion_heuristic, Karger, Solution, LargeNeighborhoodSearch, Divide, Merge, SwapNode, is_splex

S, A, W = readin('data/competition_instances/heur051_n_300_m_20122.txt')
print(S)

greedy = Karger(A, W, S)
A1, splexes = greedy.random_solution()
print('Ran greedy')
A = A.astype(np.int32)
A1 = A1.astype(np.int32)
W = W.astype(np.int32)
x0 =  Solution.build(A,W,A1, splexes) 

lns = LargeNeighborhoodSearch(
    s = S,
    destroyers = [Divide(), Merge(), SwapNode(x0.size)],
    repairers = [insertion_heuristic, deletion_heuristic],
    epochs = 100, 
    temperature = 1e4,
    alpha = 0.92,
    gamma=0.9, 
    stop_temp=10
)

x1 = lns.search(x0)

# print(x1)
print(is_splex(x1.A1, S), x1.obj())
writeout(A, x1.A1, 'output\\competition\\', 'heur051_n_300_m_20122', addendum='_'+str(x1.obj()))