import os
from funcs import *
from funcs.GRASP import GRASP
from funcs.Greedy import Karger
import frigidum
from funcs.readin import readin, writeout

# folder = 'data/test_instances/'
folder = 'data/inst_tuning/'
start = False

for file in os.listdir(folder):
    if not os.path.isfile(folder + file):
        continue
    if file == 'heur048_n_300_m_14666.txt':
        start = True
    if not start:
        continue
    prob_name = file.split('.')[0]
    print(prob_name)
    S,A,W = readin(folder + file)
    greedy = Karger(A, W, S)
    def random_start():
        A1, splexes = greedy.solution()
        print('Ran greedy')
        return Solution.build(A,W,A1, splexes) 

    mn = MoveNode(S, A.shape[0])
    def move_node(x):
        return mn.shaking(x)
    sn = SwapNode(A.shape[0])
    def swap_node(x):
        return sn.shaking(x)

    f = Flip1(S)
    def flip1(x):
        return f.shaking(x)
    
    d = Divide()
    def divide(x):
        return d.shaking(x)
    
    # Greedy construction
    # x0 = random_start()

    # local_search = VariableNeighborhoodDescent([SwapNode(A.shape[0])])
    # xb = local_search.search(x0)

    grasp = GRASP(sn, trials=10)
    xb = grasp.search(S, A, W)
    
    # xb, _ = frigidum.sa(random_start=random_start, 
    #         neighbours=[move_node, swap_node, flip1], 
    #         objective_function=lambda x: x.obj(), 
    #         T_start=10**5, 
    #         T_stop=1e-3, 
    #         repeats=10**4//2, 
    #         copy_state=frigidum.annealing.naked)
    # grasp = GRASP(SwapNode(A.shape[0]), trials=100)
    # xb = grasp.search(S,A,W)

    assert(is_splex(xb.A1, S))
    print(xb.obj())

    details = f'_grasp_{xb.obj()}'

    writeout(A, xb.A1, folder, prob_name, details)