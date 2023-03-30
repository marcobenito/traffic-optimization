from time import time
import numpy as np
import matplotlib.pyplot as plt
import random

import nasch
import genetic_algorithm as ga


# Problem definition

n = 500            #number of space positions
t = 500           #number of space intervals
density = 0.35     #density of cars in the road
P = 0.3             #Random brake probability
vmax = 5            #Maximum road speed
ntl = 3            #number of traffic lights in the road


# The entries and exits arrays must be defined. They are as follows:
#     First row: position of the entry/exit
#     Second row: Probability that a car enters os exits through this entry/exit
#     Third row: Only applies for entries. Number of cars accumulated waiting for entering
entrance_positions = np.array([120, 340],dtype=np.int64)
entrance_probability = np.array([0.15, 0.07])
exit_positions = np.array([50, 210, 430],dtype=np.int64)
exit_probability = np.array([0.08, 0.1, 0.07])
entrance = np.vstack((entrance_positions, entrance_probability, np.zeros(entrance_positions.size)))
exit = np.vstack((exit_positions, exit_probability, np.zeros(exit_positions.size)))

A = nasch.initial_scenario(t, n, density)
ITL = nasch.initialize_trafficlight(ntl,n)
#Next, the space-time diagram and fuel consumption are obtained
mean_speed = mean_cons = 0
k = 30
for i in range(k):
    A = nasch.initial_scenario(t, n, density)
    ITL = nasch.initialize_trafficlight(ntl, n)
    means = nasch.roundabout(A, P, vmax, ITL,entrance, exit)[1]
    mean_speed += means[0]
    mean_cons += means[1]
mean_speed /= k
mean_cons /= k

print('Mean Speed: ', mean_speed,'Mean consumption: ', mean_cons)

                                #########################
                                ### GENETIC ALGORITHM ###
                                #########################

n_genes = ntl * 2
n_individuals = 50
n_generations = 50
n_parents = 25
n_children = n_individuals - n_parents

def create_population(n):
    population = []
    for _ in range(n):
        individual = random.sample(range(60,250), n_genes)
        population.append(individual)

    return population

population = create_population(n_individuals)

def fitness_function(individual):
    alpha = 0.65
    betha = 1 - alpha
    # Rows 4 and 5 of the ITL must be set to the individual
    ITL = nasch.initialize_trafficlight(ntl, n)
    A = nasch.initial_scenario(t, n, density)
    ITL[1,:] = 0
    ITL[3,:] = individual[:ITL.shape[1]]
    ITL[4,:] = individual[ITL.shape[1]:]
    U, means = nasch.roundabout(A, P, vmax, ITL, entrance, exit)
    velocity = means[0]
    consumption = means[1]

    velocity_fitness = (velocity/mean_speed - 1)*alpha
    consumption_fitness = (1 - consumption/mean_cons)*betha
    fitness = velocity_fitness + consumption_fitness

    # velocity_fitness = velocity / mean_speed
    # consumption_fitness = consumption / mean_cons
    # fitness = velocity_fitness**3 / consumption_fitness

    return fitness


def ga_optimization():
    select = ga.selection()
    cross = ga.crossover()
    mutate = ga.mutation()
    # First row of best marks corresponds to the best fitness in each
    # generation and second row to its standard deviation
    best_marks = np.zeros((2, n_generations))
    best_individuals = np.zeros((n_genes,n_generations), dtype=np.int32)
    best_fitness = -99
    best_individual = []
    for i in range(n_generations):
        print('Generation: ',i)
        # Fitness calculation. It's computed several times for obtaining a
        # mean value
        reps = 5
        fitness_hut = np.zeros((reps, n_individuals))

        for j in range(reps):
            fitness_hut[j,:] = [fitness_function(ind) for ind in population]

        fitness = np.mean(fitness_hut, axis=0).tolist()
        std = np.std(fitness_hut, axis=0).tolist()
        # Once the fitness of all individuals is calculated, the survivors are
        # selected
        parents = select('sel_best', population, n_parents, fitness)
        children_after_crossover = cross('cross_blend_modified', parents, n_children, 0.5)
        children_after_mutation = mutate('mut_random_n_bit',
                                              children_after_crossover,
                                              0.5,60,250,0.4)
        population[:n_parents] = parents
        population[n_parents:] = children_after_mutation

        # Values to be stored
        # Maximum fitness and its std are stored
        best_marks[0,i] = max(fitness)
        best_marks[1,i] = std[fitness.index(best_marks[0,i])]
        # Also, the individual with the highest fitness is stored in each generation
        best_individuals[:,i] = population[fitness.index(best_marks[0,i])]
        # Global maximum fitness and best individual are stored
        if best_marks[0,i] > best_fitness:
            best_fitness = best_marks[0,i]
            best_individual = population[fitness.index(best_fitness)]

    return best_individual, best_marks, best_individuals
it = time()
best_times, best_marks, best_inds = ga_optimization1()
ft = time()
print('Time: {}'.format(ft-it))
# The chosen times will be the most common ones since the convergence point
# which is more or less n_generations / 2
final_times = np.zeros(best_inds.shape[0], dtype=np.int32)
from collections import Counter
for kk in range(best_inds.shape[0]):
    final_times[kk] = Counter(best_inds[kk,int(n_generations/2.2):]).most_common(1)[0][0]

print('Best times: {}; Fitness: {:.4f}'.format(final_times, max(best_marks[0,:])))

x = np.arange(len(best_marks[0])) + 1
fit = best_marks[0, :]
std = best_marks[1, :]
plt.figure(1)
plt.plot(x, fit)
plt.xlabel('generation')
plt.ylabel('Fitness')
plt.show()

np.save('exponencial', np.vstack((x, fit)))

new_mean_speed, new_mean_consumption = 0, 0
k1 = 1
for _ in range(k1):

    ITL = nasch.initialize_trafficlight(ntl, n)
    A = nasch.initial_scenario(t, n, density)
    ITL[1,:] = 0
    ITL[3, :] = final_times[:ITL.shape[1]]
    ITL[4, :] = final_times[ITL.shape[1]:]
    U, means = nasch.roundabout(A, P, vmax, ITL, entrance, exit)
    # U, means = nasch.straightroad(A, ITL, n, t, density, P, P, vmax, flag_entry=False)
    # print(means)
    # print(means[0])
    new_mean_speed += means[0]
    new_mean_consumption += means[1]
    # ITL[1, :] = 1
    # ITL[2, :] = 0
new_mean_speed /= k1
new_mean_consumption /= k1

print((new_mean_speed, new_mean_consumption))


velocity_fitness = new_mean_speed/mean_speed
consumption_fitness = new_mean_consumption/mean_cons
print('Speed gain: ',velocity_fitness )
print('Consumption gain: ', consumption_fitness)


A = U
for i in range(t):
    for j in range(n):
        A[i, j] = False if A[i, j] <= -1 else True

x, y = np.argwhere(A == True).T
plt.figure(1, figsize=(8, 8))
plt.scatter(y, x, s=0.01, c="blue")
plt.axis([0, n, t, 0])
plt.xlabel('Espacio')
plt.ylabel('Tiempo')
plt.show()
