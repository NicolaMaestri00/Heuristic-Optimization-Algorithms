import numpy as np

from .Solution import Solution
from .neighborhood import Neighborhood
from .VND import VariableNeighborhoodDescent
from .Greedy import GreedySPlex, Karger
import numpy.typing as npt
class GRASP:
    def __init__(self, neighborhood: Neighborhood, trials = 100) -> None:
        self.neighborhood = neighborhood
        self.local_search = VariableNeighborhoodDescent([neighborhood])
        self.trials = trials
        self.solutions_found = list()
        self.history = self.solutions_found


    def search(self, s:int, A: npt.NDArray[np.int_], W: npt.NDArray[np.int_]) -> Solution:
        greedy_search = Karger(A, W, s=s)
        
        solutions = [self.grasp_instance(greedy_search, A, W) for _ in range(self.trials)]
        self.solutions_found.extend(solutions)
        self.history = [x.obj() for x in self.solutions_found]

        return min(self.solutions_found, key=lambda x: x.obj())
    
    def grasp_instance(self, greedy_search : GreedySPlex, A: npt.NDArray[np.int_], W: npt.NDArray[np.int_]):
        A1, splexes = greedy_search.random_solution()

        x0 = Solution.build(A,W,A1,splexes)
        x1 = self.local_search.search(x0)
        print('GRASP instance, objective value:', x1.obj(),'\tStarting solution:', x0.obj())
        return x1