import numpy as np

from .Solution import Solution
from .neighborhood import Neighborhood

class SimulatedAnnealing:
    def __init__(self, init_temperature: float, neighborhood : Neighborhood, cooling : float = 0.95 ) -> None:
        self.T0 = init_temperature
        self.cooling = cooling
        self.neighborhood = neighborhood

        self.history = []

    def search(self, x0:Solution) -> Solution:
        T = self.T0
        x = x0
        best = x

        epoch = 0
        while not self.stop_criterion(epoches=epoch, temperature=T):
            iteration = 0
            while not self.equilibrium(iteration):
                x1 = self.neighborhood.shaking(x)

                p = np.exp((x.obj() - x1.obj())/T)

                if p>=1 or p > np.random.random():
                    # print(p)
                    # print(f'Found {len(self.history)}th improvement with {p:.2f}. Delta={x.obj()-x1.obj()}')
                    x = x1
                    if x.obj() < best.obj():
                        best = x
                self.history.append(x.obj())

                iteration+=1
            
            epoch+=1
            T = T*self.cooling
        return best

                

    def equilibrium(self, iteration : int, step : int = 5) -> bool:
        return iteration > 2
        if iteration < 10:
            return False
        h = self.history[-iteration:]
        rho = np.corrcoef(h[:-step], h[step:])[0,1] 
        print(rho)
        return rho > 0.99
        
    def stop_criterion(self, epoches : int, temperature:float, max_epoches : int = 20) -> bool:
        return temperature < 1e-4