from funcs import readin, Solution, GA_new_version, is_splex
from funcs.Greedy import Karger
import numpy as np
from funcs.readin import writeout
import os


# Tuning istances
# file = 'data/inst_tuning/heur040_n_300_m_13358.txt'
# file = 'data/inst_tuning/heur041_n_300_m_17492.txt'
# file = 'data/inst_tuning/heur042_n_300_m_5764.txt'
# file = 'data/inst_tuning/heur043_n_300_m_12914.txt'
# file = 'data/inst_tuning/heur058_n_300_m_4010.txt'          # OK
# file = 'data/inst_tuning/heur059_n_300_m_7867.txt'            # OK

# S,A,W = readin(file)

folder = 'data/inst_tuning/'

for file in os.listdir(folder):
    if not os.path.isfile(folder + file):
        continue
    if file != 'heur060_n_300_m_12405.txt':
        continue
    print(file)
    S,A,W = readin(folder + file)
    # mean_fitness_population = []
    # best_fitness_population = []
    # for i in range(5):
    #     # Genetic Algorithm
    #     GeneticAlgorithm = GA_new_version.GeneticAlgorithm_modified(A, W, S, length_population=100, k=0.5)
    #     final_population = GeneticAlgorithm.evolution(3000)

    #     for i in range(len(final_population)):
    #         if  not is_splex(final_population[i].A1, S):
    #             print('Attention: There\'s an Error!!!')
    #             break

    #     mean_fitness_population.append(np.mean([solution.obj() for solution in final_population]))
    #     best_fitness_population.append(min([solution.obj() for solution in final_population]))
    # print(f'Best fitness: {np.mean(best_fitness_population)}')
    # print(best_fitness_population)
    # print(f'Mean fitness: {np.mean(mean_fitness_population)}')
    # print(mean_fitness_population)

    # Genetic Algorithm

    GeneticAlgorithm = GA_new_version.GeneticAlgorithm_modified(A, W, S, length_population=100, k=1)
    final_population, mean_fitness_population, best_fitness_population = GeneticAlgorithm.evolution(3000)
    for i in range(len(final_population)):
        if  not is_splex(final_population[i].A1, S):
            print('Attention: There\'s an Error!!!')
            break

print('Mean fitness:')
print(mean_fitness_population)
print('Best fitness:')
print(best_fitness_population)  

# 10 time Tuning test
# mean_fitness_population = []
# best_fitness_population = []
# for i in range(10):
#     # Genetic Algorithm
#     GeneticAlgorithm = GA_new_version.GeneticAlgorithm_modified(A, W, S, length_population=100, k=0.0)
#     final_population = GeneticAlgorithm.evolution(3000)

#     for i in range(len(final_population)):
#         if  not is_splex(final_population[i].A1, S):
#             print('Attention: There\'s an Error!!!')
#             break

#     mean_fitness_population.append(np.mean([solution.obj() for solution in final_population]))
#     best_fitness_population.append(min([solution.obj() for solution in final_population]))


# Genetic Algorithm
# GeneticAlgorithm = GA_new_version.GeneticAlgorithm_modified(A, W, S, length_population=100, k=0.0)
# final_population = GeneticAlgorithm.evolution(3000)

# for i in range(len(final_population)):
#     if  not is_splex(final_population[i].A1, S):
#         print('Attention: There\'s an Error!!!')
#         break

# folder = 'data/competition_results'

# xb = min(final_population, key=lambda x: x.obj())
# details = f'_GA_{xb.obj()}'
# prob_name = 'heur049_n_300_m_17695'
# prob_name = 'heur050_n_300_m_19207'
# prob_name = 'heur051_n_300_m_20122'
# if is_splex(xb.A1, S)==True:
#     print('Admissible')
# print(xb.obj())
# writeout(A, xb.A1, folder, prob_name, details)
