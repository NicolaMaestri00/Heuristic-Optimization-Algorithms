import os
from funcs import *
import numpy as np
from funcs import readin, insertion_heuristic, deletion_heuristic, Karger, Solution, LargeNeighborhoodSearch, Divide, Merge, SwapNode, is_splex
from itertools import product
from tqdm import tqdm
def grid(inp):
    return [dict(zip(inp.keys(), values)) for values in product(*inp.values())]


folder = 'data/tuning_instances/'
# folder = 'data/competition_instances/'
record = []

params = grid({
    'temperature': [1e2, 1e3, 1e4],
    'alpha' : [0.92, 0.95],
    'gamma' : [0.7, 0.9], 
    'epochs' : [100]
})


for file in os.listdir(folder):
    if not os.path.isfile(folder + file):
        continue
    prob_name = file.split('.')[0]
    print(prob_name)
    S,A,W = readin(folder + file)
    greedy = Karger(A, W, S)
    A1, splexes = greedy.random_solution()
    print('Ran greedy')
    A = A.astype(np.int32)
    A1 = A1.astype(np.int32)
    W = W.astype(np.int32)
    x0 =  Solution.build(A,W,A1, splexes)

    for param in tqdm(params):
        lns = LargeNeighborhoodSearch(
            s = S,
            destroyers = [Divide(), Merge(), SwapNode(x0.size)],
            repairers = [insertion_heuristic, deletion_heuristic],
            epochs = param['epochs'], 
            temperature = param['temperature'],
            alpha = param['alpha'],
            gamma = param['gamma']
        )

        x1 = lns.search(x0)
        assert is_splex(x1.A1, S)
        x1._obj = -1
        # print(x1.obj())
        record.append((prob_name, param['epochs'], param['temperature'], param['alpha'], param['gamma'], x1.obj()))


import pandas as pd 
df = pd.DataFrame.from_records(record, columns=['name', 'epochs', 'temperature', 'alpha', 'gamma', 'objective'])
df.to_csv('output/grid_search_11.csv', index=False)


    