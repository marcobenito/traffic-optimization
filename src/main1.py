# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:15:20 2019

@author: marco
"""
from dask import delayed
from time import time
from timeit import timeit
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import sys
import random
import multiprocessing as mp
from joblib import Parallel, delayed
from numba import njit, jit, prange
from numba.dispatcher import Dispatcher
from numba.errors import NumbaWarning, NumbaDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import nasch
import genetic_algorithm as ga


# Problem definition

n = 150            #number of space positions
t = 500           #number of space intervals
density = 0.7    #density of cars in the road
P = 0.3             #Random brake probability
vmax = 5            #Maximum road speed
ntl = 2             #number of traffic lights in the road


# The entries and exits arrays must be defined. They are as follows:
#     First row: position of the entry/exit
#     Second row: Probability that a car enters os exits through this entry/exit
#     Third row: Only applies for entries. Number of cars accumulated waiting for entering
entrance_positions = np.array([120, 540],dtype=np.int64)
entrance_probability = np.array([0, 0],dtype=np.int64)
exit_positions = np.array([50, 210, 430],dtype=np.int64)
exit_probability = np.array([0, 0, 0],dtype=np.int64)

entrance = np.vstack((entrance_positions, entrance_probability, np.zeros(entrance_positions.size)))
exit = np.vstack((exit_positions, exit_probability, np.zeros(exit_positions.size)))



A = nasch.initial_scenario(t, n, density)
ITL = nasch.initialize_trafficlight(ntl,n)
#Next, the space-time diagram and fuel consumption are obtained
# A, means = nasch.roundabout(A, ITL,entrance, exit, P, vmax)
# A, means = nasch.straightroad(A, ITL, n, t, density, P, vmax, flag_entry = True)
mean_speed = mean_cons = 0
k = 10
for i in range(k):
    A = nasch.initial_scenario(t, n, density)
    ITL = nasch.initialize_trafficlight(ntl, n)
    # means = nasch.roundabout(A, P, vmax, ITL,entrance, exit)[1]
    A, means = nasch.straightroad(A, ITL, n, t, density, P, P, vmax, flag_entry=True)
    mean_speed += means[0]
    mean_cons += means[1]
mean_speed /= k
mean_cons /= k
for i in range(t):
    for j in range(n):
        A[i,j] = False if A[i,j] <=-1 else True

x,y = np.argwhere(A == True).T
plt.figure(1, figsize=(5,6))
plt.scatter(y,x,s=0.01,c="blue")
plt.axis([0,n,t, 0])
plt.xlabel('Espacio')
plt.ylabel('Tiempo')

plt.show()
print(ITL)
print('Mean Speed: ', mean_speed,'Mean consumption: ', mean_cons)


def fitness_function_sr(individual):
    alpha = 0.95
    betha = 1 - alpha
    # Rows 4 and 5 of the ITL must be set to the individual
    ITL = nasch.initialize_trafficlight(ntl, n)
    A = nasch.initial_scenario(t, n, density)
    ITL[1, :] = 0
    ITL[3, :] = individual
    ITL[4, :] = 1E7
    U, means = nasch.straightroad(A, ITL, n, t, density, P, P,vmax, entrance, exit, flag_entry=True)
    velocity = means[0]
    consumption = means[1]
    fitness = (velocity / mean_speed - 1) * alpha + (1 - consumption / mean_cons) * betha
    ad_vel = velocity / mean_speed
    ad_cons = consumption / mean_cons
    fitness = ad_vel / ad_cons

    return fitness

# @jit(cache=True)
# def fitness_function(individual, A, ITL, entrance, exit, P, vmax):
def fitness_function(individual, A, P, vmax, ITL, entrance=None, exit=None, alpha=0.5, betha=0.5):
# def fitness_function(individual):
    alpha = 0.975
    betha = 1 - alpha
    # alpha = 1.5
    # betha = 0.875*alpha + 0.375
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
    fitness = (velocity/mean_speed - 1)*alpha + (1 - consumption/mean_cons)*betha
    # fitness = ((velocity / mean_speed) ** alpha) * ((mean_cons / consumption) ** betha)
    # velocity_fitness = (velocity_fitness/mean_speed - 1)
    # consumption_fitness = (1 - consumption_fitness/mean_cons)
    # return (fitness,)
    return fitness #[round(velocity_fitness,3), round(consumption_fitness,3)]#


n_genes = ntl * 1
n_individuals = 50
n_generations =50
n_parents = 25
n_children = n_individuals - n_parents

def create_population(n):
    population = []
    for _ in range(n):
        individual = random.sample(range(0,500), n_genes)
        population.append(individual)

    return population

population = create_population(n_individuals)


def return_fitness(population):
    return [fitness_function(ind) for ind in population]


def parents_fitness(fit, pop, parents, std=None):
    idxs = [pop.index(parent) for parent in parents]
    parents_fit = [fit[idx] for idx in idxs]
    if std is not None:
        parents_std = [std[idx] for idx in idxs]
        return parents_fit, parents_std
    else:
        return parents_fit


# population = np.asarray(population).transpose()

# population_size = (ntl * 2, 50)
# population = np.random.uniform(50, 400, population_size)
# population = population
n_cores = mp.cpu_count()

# For straightroad
def ga_optimization_sr():
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
        reps = 10
        fitness_hut = np.zeros((reps, n_individuals))

        for j in range(reps):
            # for ind in range(n_individuals):
            #     fitness_hut[j,ind] = fitness_function_sr(population[ind], A, P, vmax, ITL, entrance, exit)
            fitness_hut[j,:] = [fitness_function_sr(ind) for ind in population]

        fitness = np.mean(fitness_hut, axis=0).tolist()
        std = np.std(fitness_hut, axis=0).tolist()
        # Once the fitness of all individuals is calculated, the survivors are
        # selected
        parents = select('sel_best', population, n_parents, fitness)
        children_after_crossover = cross('cross_blend_modified', parents, n_children, 0.5)
        children_after_mutation = mutate('mut_gaussian',
                                              children_after_crossover,
                                              0.2,0,10,0.4)
        population[:n_parents] = parents
        population[n_parents:] = children_after_mutation

        best_marks[0,i] = max(fitness)
        best_marks[1,i] = std[fitness.index(best_marks[0,i])]
        best_individuals[:,i] = population[fitness.index(best_marks[0,i])]
        if best_marks[0,i] > best_fitness:
            best_fitness = best_marks[0,i]
            best_individual = population[fitness.index(best_fitness)]

    # best_individual = population[fitness.index(max(fitness))]
    return best_individual, best_marks, best_individuals


# Normal version with lists
# @jit
# @jit(cache=True, parallel=True, nogil=True)
# @delayed
def ga_optimization1():
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
        reps = 8
        fitness_hut = np.zeros((reps, n_individuals))

        for j in range(reps):
            for ind in range(n_individuals):
                fitness_hut[j,ind] = fitness_function(population[ind], A, P, vmax, ITL, entrance, exit)
            # fitness_hut[j,:] = [fitness_function(ind, A, P, vmax, ITL, entrance, exit) for ind in population]

        fitness = np.mean(fitness_hut, axis=0).tolist()
        std = np.std(fitness_hut, axis=0)
        # Once the fitness of all individuals is calculated, the survivors are
        # selected
        parents = select('sel_best', population, n_parents, fitness)
        children_after_crossover = cross('cross_blend_modified', parents, n_children, 0.5)
        children_after_mutation = mutate('mut_random_n_bit',
                                              children_after_crossover,
                                              1,40,500,0.4)
        population[:n_parents] = parents
        population[n_parents:] = children_after_mutation

        best_marks[0,i] = max(fitness)
        best_marks[1,i] = std[fitness.index(best_marks[0,i])]
        best_individuals[:,i] = population[fitness.index(best_marks[0,i])]
        if best_marks[0,i] > best_fitness:
            best_fitness = best_marks[0,i]
            best_individual = population[fitness.index(best_fitness)]

    # best_individual = population[fitness.index(max(fitness))]
    return best_individual, best_marks, best_individuals

# Normal lists version with reduced fitness calculation
def ga_optimization10():
    select = ga.selection()
    cross = ga.crossover()
    mutate = ga.mutation()
    # First row of best marks corresponds to the best fitness in each
    # generation and second row to its standard deviation
    best_marks = np.zeros((2, n_generations))
    best_individuals = np.zeros((n_genes,n_generations), dtype=np.int32)
    best_fitness = -99
    best_individual = []

    # First generation fitness calculation
    reps = 1
    fitness_hut = np.zeros((reps, n_individuals))

    for j in range(reps):
        for ind in range(n_individuals):
            fitness_hut[j, ind] = fitness_function(population[ind], A, P, vmax, ITL, entrance, exit)
        # fitness_hut[j,:] = [fitness_function(ind, A, P, vmax, ITL, entrance, exit) for ind in population]

    fitness = np.mean(fitness_hut, axis=0).tolist()
    std = np.std(fitness_hut, axis=0).tolist()


    for i in range(n_generations):
        print('Generation: ',i)
        # Fitness calculation. It's computed several times for obtaining a
        # mean value

        # Once the fitness of all individuals is calculated, the survivors are
        # selected
        parents = select('sel_best', population, n_parents, fitness)
        # Fitness of parents is stored for next generation
        fitness_parents, std_parents = parents_fitness(fitness, population, parents, std)

        children_after_crossover = cross('cross_one_child', parents, n_children, 0.5)
        children_after_mutation = mutate('mut_random_n_bit',
                                              children_after_crossover,
                                              1,0,500,0.4)
        population[:n_parents] = parents
        population[n_parents:] = children_after_mutation
        best_marks[0,i] = max(fitness)
        best_marks[1,i] = std[fitness.index(best_marks[0,i])]
        best_individuals[:,i] = population[fitness.index(best_marks[0,i])]
        if best_marks[0,i] > best_fitness:
            best_fitness = best_marks[0,i]
            best_individual = population[fitness.index(best_fitness)]

        # Fitness of the next generation is calculated
        fitness_hut = np.zeros((reps, n_children))

        for j in range(reps):
            for ind in range(n_children):
                fitness_hut[j, ind] = fitness_function(children_after_mutation[ind], A, P, vmax, ITL, entrance, exit)
            # fitness_hut[j,:] = [fitness_function(ind, A, P, vmax, ITL, entrance, exit) for ind in population]

        fitness_children = np.mean(fitness_hut, axis=0).tolist()
        std_children = np.std(fitness_hut, axis=0).tolist()

        fitness = fitness_parents + fitness_children
        std = std_parents + std_children

    # best_individual = population[fitness.index(max(fitness))]
    return best_individual, best_marks, best_individuals

# Normal lists version with parallelization
def ga_optimization11():
    select = ga.selection()
    cross = ga.crossover()
    mutate = ga.mutation()
    # First row of best marks corresponds to the best fitness in each
    # generation and second row to its standard deviation
    best_marks = np.zeros((2, n_generations))
    num_cores = mp.cpu_count()

    for i in range(n_generations):
        print('Generation: ',i)
        # Fitness calculation. It's computed several times for obtaining a
        # mean value
        reps = 1
        fitness_hut = np.zeros((reps, n_individuals))
        with Parallel(n_jobs=num_cores) as parallel:
            a = 0.65
            b = 0.35
            fitness_hut = parallel(delayed(fitness_function)
                                           (population[ind],A, P, vmax, ITL, entrance, exit, a, b)
                                           for j in range(reps) for ind in range(n_individuals))
        # for j in range(reps):
        #     for ind in range(n_individuals):
        #         fitness_hut[j,ind] = fitness_function(population[ind], A, P, vmax, ITL, entrance, exit)

        fitness = np.mean(fitness_hut, axis=0).tolist()
        std = np.std(fitness_hut, axis=0)
        # Once the fitness of all individuals is calculated, the survivors are
        # selected
        parents = select('sel_best', population, n_parents, fitness)
        children_after_crossover = cross('cross_one_point', parents, n_children, 0.5)
        children_after_mutation = mutate('mut_random_one_bit',
                                              children_after_crossover,
                                              1, 60,250)
        population[:n_parents] = parents
        population[n_parents:] = children_after_mutation
        best_marks[0,i] = max(fitness)
        best_marks[1,i] = std[fitness.index(best_marks[0,i])]

    best_individual = population[fitness.index(max(fitness))]
    return best_individual, best_marks

# Lists version for checking fitness function
def ga_optimization2(a, b):
    select = ga.selection()
    cross = ga.crossover()
    mutate = ga.mutation()
    # First row of best marks corresponds to the best fitness in each
    # generation and second row to its standard deviation
    best_marks = np.zeros((2, n_generations))
    best_fitness = -99
    best_individual = []
    for i in range(n_generations):
        # print('Generation: ',i)
        # Fitness calculation. It's computed several times for obtaining a
        # mean value
        reps = 6
        fitness_hut = np.zeros((reps, n_individuals))
        for j in range(reps):
            fitness_hut[j,:] = [fitness_function(ind, A, P, vmax, ITL, entrance, exit, a, b) for ind in population]
        fitness = np.mean(fitness_hut, axis=0).tolist()
        std = np.std(fitness_hut, axis=0)
        # Once the fitness of all individuals is calculated, the survivors are
        # selected
        parents = select('sel_best', population, n_parents, fitness)
        children_after_crossover = cross('cross_one_child', parents, n_children, 0.5)
        children_after_mutation = mutate('mut_random_one_bit',
                                              children_after_crossover,
                                              1, 0,500)
        population[:n_parents] = parents
        population[n_parents:] = children_after_mutation
        best_marks[0, i] = max(fitness)
        best_marks[1, i] = std[fitness.index(best_marks[0, i])]
        if best_marks[0, i] > best_fitness:
            best_fitness = best_marks[0, i]
            best_individual = population[fitness.index(best_fitness)]

    # best_individual = population[fitness.index(max(fitness))]
    return best_individual, best_marks

# Lists version for checking fitness function with reduced fitness
def ga_optimization20(a, b):
    select = ga.selection()
    cross = ga.crossover()
    mutate = ga.mutation()
    # First row of best marks corresponds to the best fitness in each
    # generation and second row to its standard deviation
    best_marks = np.zeros((2, n_generations))
    best_fitness = -99
    best_individual = []

    # First generation fitness calculation
    reps = 6
    fitness_hut = np.zeros((reps, n_individuals))

    for j in range(reps):
        for ind in range(n_individuals):
            fitness_hut[j, ind] = fitness_function(population[ind], A, P, vmax, ITL, entrance, exit, a, b)
        # fitness_hut[j,:] = [fitness_function(ind, A, P, vmax, ITL, entrance, exit) for ind in population]

    fitness = np.mean(fitness_hut, axis=0).tolist()
    std = np.std(fitness_hut, axis=0).tolist()

    for i in range(n_generations):
        # print('Generation: ',i)
        # Fitness calculation. It's computed several times for obtaining a
        # mean value

        # Once the fitness of all individuals is calculated, the survivors are
        # selected
        parents = select('sel_best', population, n_parents, fitness)
        # Fitness of parents is stored for next generation
        fitness_parents, std_parents = parents_fitness(fitness, population, parents, std)

        children_after_crossover = cross('cross_one_child', parents, n_children, 0.5)
        children_after_mutation = mutate('mut_random_one_bit',
                                              children_after_crossover,
                                              1, 0,500)
        population[:n_parents] = parents
        population[n_parents:] = children_after_mutation
        best_marks[0, i] = max(fitness)
        best_marks[1, i] = std[fitness.index(best_marks[0, i])]
        if best_marks[0, i] > best_fitness:
            best_fitness = best_marks[0, i]
            best_individual = population[fitness.index(best_fitness)]

        # Fitness of the next generation is calculated
        fitness_hut = np.zeros((reps, n_children))

        for j in range(reps):
            for ind in range(n_children):
                fitness_hut[j, ind] = fitness_function(children_after_mutation[ind], A, P, vmax, ITL, entrance,
                                                       exit, a, b)
            # fitness_hut[j,:] = [fitness_function(ind, A, P, vmax, ITL, entrance, exit) for ind in population]

        fitness_children = np.mean(fitness_hut, axis=0).tolist()
        std_children = np.std(fitness_hut, axis=0).tolist()

        fitness = fitness_parents + fitness_children
        std = std_parents + std_children

    # best_individual = population[fitness.index(max(fitness))]
    return best_individual, best_marks

# Lists version for cheking population size
def ga_optimization3(population):
    select = ga.selection()
    cross = ga.crossover()
    mutate = ga.mutation()
    # First row of best marks corresponds to the best fitness in each
    # generation and second row to its standard deviation
    best_marks = np.zeros((2, n_generations))
    print('Pop length: ', len(population))
    for i in range(n_generations):
        print('Generation: ',i)
        # Fitness calculation. It's computed several times for obtaining a
        # mean value
        reps = 4
        fitness_hut = np.zeros((reps, len(population)))
        for j in range(reps):
            fitness_hut[j,:] = [fitness_function(ind, A, P, vmax, ITL, entrance, exit) for ind in population]
        fitness = np.mean(fitness_hut, axis=0).tolist()
        std = np.std(fitness_hut, axis=0)
        # Once the fitness of all individuals is calculated, the survivors are
        # selected
        parents = select('sel_best', population, n_parents, fitness)
        children_after_crossover = cross('cross_one_child', parents, n_children, 0.5)
        children_after_mutation = mutate('mut_random_one_bit',
                                              children_after_crossover,
                                              1, 0,500)
        population[:n_parents] = parents
        population[n_parents:] = children_after_mutation
        best_marks[0,i] = max(fitness)
        best_marks[1,i] = std[fitness.index(best_marks[0,i])]

    best_individual = population[fitness.index(max(fitness))]
    return best_individual, best_marks
# Lists version for lexicase selection
def ga_optimization4():
    select = ga.selection()
    cross = ga.crossover()
    mutate = ga.mutation()
    # First row of best marks corresponds to the best fitness in each
    # generation and second row to its standard deviation
    best_marks = np.zeros((2, n_generations))
    for i in range(n_generations):
        print('Generation: ',i)
        # Fitness calculation. It's computed several times for obtaining a
        # mean value
        reps = 10
        fitness_hut = np.zeros((reps, n_individuals))
        fitness_hut_2 = np.zeros((reps, n_individuals))

        for j in range(reps):
            fitness = [fitness_function(ind, A, P, vmax, ITL, entrance, exit) for ind in population]
            # print(len(fitness))
            fitness_hut[j,:] = [(fitness[ii][0]) + (fitness[ii][1]) for ii in range(len(fitness))]
            # fitness_hut_2[j,:] = fitness[1]
        # Once the fitness of all individuals is calculated, the survivors are
        # selected
        used_fitness = np.mean(fitness_hut, axis=0).tolist()
        std = np.std(fitness_hut, axis=0)
        parents = select('sel_lexicase', population, n_parents, fitness, weights=[1,-1])
        children_after_crossover = cross('cross_one_child', parents, n_children, 0.5)
        children_after_mutation = mutate('mut_random_one_bit',
                                              children_after_crossover,
                                              1, 0,500)
        population[:n_parents] = parents
        population[n_parents:] = children_after_mutation
        # vel_fit = [fit[0] for fit in fitness]
        # best_marks[0,i] = max(vel_fit)
        # best_marks[1, i] = std[vel_fit.index(best_marks[0, i])]

        # cons_fit = [fit[1] for fit in fitness]
        # fitness = [(vel_fit[i] - 1)*0.65 + (1-cons_fit[i])*0.35 for i in range(len(vel_fit))]
        best_marks[0,i] = max(used_fitness)
        best_marks[1,i] = std[used_fitness.index(best_marks[0,i])]
    best_individual = population[used_fitness.index(max(used_fitness))]
    # best_individual = population[fitness.index(max(fitness))]
    return best_individual, best_marks

# @njit
# Original version with arrays
def ga_optimization():
    best_marks = np.zeros(n_generations)

    for i in range(n_generations):
        print('Generacion: ', i)
        # fitness = GA.pop_fitness(population, 5, fitness_function)
        fitness = np.asarray([fitness_function(ind) for ind in population.transpose()])
        # fitness = np.asarray(list(map(fitness_function, population.transpose())))
        parents = GA.parents_selection(population, fitness, num_parents)
        children_after_crossover = GA.crossover(parents,
                                                children_size=(population.shape[0], population.shape[1] - parents.shape[1]))
        children_after_mutation = GA.mutation(children_after_crossover)
        population[:, 0:parents.shape[1]] = parents
        population[:, parents.shape[1]:] = children_after_mutation

        best_marks[i] = np.max(fitness)
    # print('Acabo')
    best_id = np.where(fitness == np.max(fitness))[0][0]
    best_individual = population[:, best_id]

    return best_individual, best_marks

# ♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥

# it = time()


from deap import base, creator, tools
import random
alpha_v = 0.85
alpha_c = 0.15
num_parents = 22
creator.create("FitnessTL", base.Fitness, weights=(1,))
creator.create("Individual", list, fitness=creator.FitnessTL)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 500)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual,
    toolbox.attr_bool, ntl*2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt,low=0, up=500, indpb=1/(2*ntl))
toolbox.register("select", tools.selTournament, tournsize=5)
best_sols = best_sols1 = []
def main():
    # random.seed(64)

    pop = toolbox.population(n=50)
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # CXPB  is the probability with which two individuals are crossed
    #  MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while g < 100:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, 20)#k=num_parents)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        # fits1 = [ind.fitness.values[1] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        # print("  Min %s" % min(fits))
        # print("  Max %s" % max(fits))
        # print("  Avg %s" % mean)
        # print("  Std %s" % std)

        best_sols.append(max(fits))
        # best_sols1.append(max(fits1))

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

from scipy.optimize import differential_evolution

# def main_scipy():
#     bounds = [(50,400), (50,400), (50,400), (50,400)]
#     result = differential_evolution(fitness_function, bounds, maxiter=30, popsize=50)
#     print(result.x, result.fun)
#
# it = time()
# main_scipy()
# ft = time()
# print('Tiempo de ejecucion scipy: ',ft-it)

# Checking fitness function (alpha & beta)
if __name__ == '__mai3n__':
    it = time()
    k = 11
    alpha = np.linspace(0.45, 0.6, k)
    betha = 1 - alpha
    store = []
    speed = []
    cons = []
    alphas = []
    for idx, (i, j) in enumerate(zip(alpha, betha)):
        print('Iter: ', idx + 1, 'alpha = ', i)
        best_t, best_marks = ga_optimization2(i, j)

        # store[idx] = best_marks[0,-1]
        # store.append((best_marks[0,-1]))
        # ITL[3, :] = best_t[:ITL.shape[1]]
        # ITL[4, :] = best_t[ITL.shape[1]:]
        new_mean_speed = new_mean_consumption = 0
        k1 = 10
        for _ in prange(k1):
            ITL[3, :] = best_t[:ITL.shape[1]]
            ITL[4, :] = best_t[ITL.shape[1]:]
            means = nasch.roundabout(A, P, vmax, ITL, entrance, exit)[1]
            new_mean_speed += means[0]
            new_mean_consumption += means[1]
            ITL[1, :] = 1
            ITL[2, :] = 0
        new_mean_speed /= k1
        new_mean_consumption /= k1
        print('speed: ', new_mean_speed/mean_speed)
        print('cons: ', new_mean_consumption/mean_cons)
        print('fitness: ', max(best_marks[0,:]))
        print('best times: ', best_t)
        if new_mean_speed/mean_speed > 0:
            store.append(best_marks[0,-1])
            speed.append(new_mean_speed/mean_speed)
            cons.append(new_mean_consumption/mean_cons)
            alphas.append(i)
    ft = time()
    print('Total time: ', ft-it)

    plt.figure(1)
    plt.plot(alphas, store)
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(alpha, betha, store)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Fitness')

    plt.figure(2)
    plt.plot(alphas, store, label='Fitness')
    plt.plot(alphas, speed, label='V/V*')
    plt.plot(alphas, cons, label='C/C*')
    plt.xlabel(r'$\alpha$')
    plt.legend()
    # plt.ylabel('Fitness')

    fig, ax1 = plt.subplots()
    ax1.plot(alphas, store, label='Fitness')
    ax1.set_xlabel(r'$\alpha$')
    # ax1.tick_params('x', labelrotation=90)
    ax1.set_ylabel('Fitness')
    ax2 = ax1.twinx()  # Define a new ax with the same x axis
    ax2.set_ylabel('V/V*, C/C*')
    ax2.plot(alphas, speed, label='V/V*', color='red')
    ax2.plot(alphas, cons, label='C/C*', color='green')
    fig.legend()
    fig.tight_layout()  # Otherwise, the y axis will be a little displaced
    # ax1.xaxis.set_major_locator(plt.MaxNLocator(7))  # 7 ticks in the x label at most
    # plt.title('Ocupacion y velocidad media durante todo el día 03-02-2020, sensor 6644 (km 1.1)')

    plt.show()
# Checking exponential fitness function
if __name__ == '__mai5n__':
    it = time()
    k = 12
    alpha = betha = np.linspace(1, 3, k)
    # betha = 1 - alpha
    store = np.zeros((k, k))
    speed = np.zeros((k, k))
    cons = np.zeros((k, k))
    # alphas = []
    for i, a in enumerate(alpha):
        for j, b in enumerate(betha):
            print('Iter: ' + str(k*i + j + 1) + '/' + str(k**2))

            best_t, best_marks = ga_optimization20(i, j)


            new_mean_speed = new_mean_consumption = 0
            k1 = 10
            for _ in range(k1):
                ITL[3, :] = best_t[:ITL.shape[1]]
                ITL[4, :] = best_t[ITL.shape[1]:]
                means = nasch.roundabout(A, P, vmax, ITL, entrance, exit)[1]
                new_mean_speed += means[0]
                new_mean_consumption += means[1]
                ITL[1, :] = 1
                ITL[2, :] = 0
            new_mean_speed /= k1
            new_mean_consumption /= k1

            store[i,j] = best_marks[0,-1]
            speed[i,j] = new_mean_speed/mean_speed
            cons[i,j] = new_mean_consumption/mean_cons
    ft = time()
    print('Total time: ', ft-it)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(alpha, betha)
    ax.plot_surface(X, Y, store, cmap='viridis')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    ax.set_zlabel('Fitness')
    # ax.axis([0, t, t, 0])

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(alpha, betha)
    ax1.plot_surface(X, Y, speed, cmap='viridis')
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'$\beta$')
    ax1.set_zlabel('V/V*')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(alpha, betha)
    ax2.plot_surface(X, Y, cons, cmap='viridis')
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'$\beta$')
    ax2.set_zlabel('C/C*')

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(alpha, betha)
    ax3.plot_surface(X, Y, speed/cons, cmap='viridis')
    ax3.set_xlabel(r'$\alpha$')
    ax3.set_ylabel(r'$\beta$')
    ax3.set_zlabel(r'$\frac{V/V*}{C/C*}$')

    plt.show()

# Normal (function name)
if __name__ == '__main__':
    it = time()
    function_to_analyze = 'mut_gaussian'  # reps = 25
    # best_marks_hut = np.zeros((reps, 50))
    # for i in range(reps):
    population = create_population(n_individuals)
    best_times, best_marks, best_inds = ga_optimization_sr()
    # best_marks = np.mean(best_marks_hut, axis=0)
    # std = np.std(best_marks_hut, axis=0)
    ft = time()

# best_times, best_marks = ga_optimization1()
    # The chosen times will be the most common ones since the convergence point
    # which is more or less n_generations / 2
    final_times = np.zeros(best_inds.shape[0], dtype=np.int32)
    from collections import Counter
    for kk in range(best_inds.shape[0]):
        final_times[kk] = Counter(best_inds[kk,25:]).most_common(1)[0][0]

    print('Mejores timepos: ', final_times, max(best_marks[0,:]))
    print('Tiempo de ejecucion ownGA: ',ft-it)
    # # print(timeit(main,number = 1000))
    # ITL[3,:] = best_times[:ITL.shape[1]]
    # ITL[4,:] = best_times[ITL.shape[1]:]
    new_mean_speed, new_mean_consumption = 0, 0
    k1 = 1
    for _ in range(k1):
        A = nasch.initial_scenario(1500, n, density)
        ITL = nasch.initialize_trafficlight(ntl,n)
        ITL[1, :] = 0
        ITL[3, :] = final_times
        ITL[4, :] = 1000
        # U, means = nasch.roundabout(A, P, vmax, ITL, entrance, exit)
        U, means = nasch.straightroad(A, ITL, n, t, density, P, P, vmax, flag_entry=True)
        print(means)
        print(means[0])
        new_mean_speed += means[0]
        new_mean_consumption += means[1]
        # ITL[1, :] = 1
        # ITL[2, :] = 0
    new_mean_speed /= k1
    new_mean_consumption /= k1
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

    # U, means = nasch.roundabout(A, ITL, entrance, exit, P, vmax)
    # velocity_fitness = means[0]/mean_speed
    # consumption_fitness = means[1]/mean_cons
    velocity_fitness = new_mean_speed/mean_speed
    consumption_fitness = new_mean_consumption/mean_cons
    print('Speed gain: ',velocity_fitness )
    print('Consumption gain: ', consumption_fitness)

    x = np.arange(len(best_marks[0]))+1
    fit = best_marks[0,:]
    std = best_marks[1,:]
    plt.figure(1)
    plt.plot(x, fit)
    # plt.plot(x, fit+std)
    # plt.plot(x, fit-std)

    plt.figure(2)
    plt.fill_between(x, fit+std, fit-std)

    y = savgol_filter(fit, 11, 3)
    y1 = savgol_filter(fit+std, 11, 3)
    y2 = savgol_filter(fit-std, 11, 3)
    plt.figure(3)
    plt.plot(x, y, x, y1, x, y2)

    plt.figure(4)
    plt.plot(x,y1,x,y2, color='blue', alpha=0.5)
    plt.fill_between(x, y1, y2, alpha=0.2)

    colors = [1, 0.6]
    plt.figure(5)
    for jj in range(best_inds.shape[0]):
        plt.plot(x, best_inds[jj, :], 'r', alpha=colors[jj])
    plt.xlabel('# generations')
    plt.ylabel('Time [s]')
    plt.yticks([0,100,200,300,400,500])
    plt.legend(['Red TL1', 'Red TL2'])
    # plt.legend(['Red TL1', 'Red TL2', 'Red TL3', 'Green TL1', 'Green TL2', 'Green TL3'])

    for tl in range(ntl):
        plt.figure(6+tl)
        plt.plot(x, best_inds[tl,:], color='red', label='Red TL{}'.format(tl+1))
        # plt.plot(x,best_inds[ntl+tl], color='green', label='Green TL{}'.format(tl+1))
        plt.legend()
    plt.xlabel('# generations')
    plt.ylabel('Time [s]')


    plt.show()

    print('Desea guardar el resultado? (y/n))')
    g = input()
    if g == 'y':
        print('Guardado')
        path = '../dat/saved_for_plotting/'
        file = path + function_to_analyze
        np.save(file, best_marks)
#Cheking population size
if __name__ == '__main22__':
    it = time()
    pop_size = [10, 50, 100, 500, 1000]
    best_marks1 = [[]]
    times = []
    plt.figure()
    for n in pop_size:
        it = time()
        population = create_population(n)
        best_t, best_marks = ga_optimization3(population)
        ft = time()
        times.append(ft-it)

        x = np.arange(len(best_marks[0])) + 1
        fit = best_marks[0, :]
        y = savgol_filter(fit, 11, 3)
        plt.plot(x, y, label='Pop size: ' + str(n))

    plt.xlabel('# iterations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()

    plt.figure()
    plt.scatter(times, pop_size)
    plt.xlabel('Time [s]')
    plt.ylabel('Population size')
    plt.show()

    ft = time()
    # print('Total time: ', ft-it)

    plt.figure()
    plt.plot(alpha, store)
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(alpha, betha, store)
    plt.show()


# DEAP Version
if __name__ == "__main3__":
    it = time()
    main()
    ft = time()
    print('Tiempo de ejecucion DEAP: ', ft-it)

    plt.figure()
    plt.plot(range(len(best_sols)), best_sols, label = 'Speed')
    # # # # plt.plot(range(len(best_sols1)), best_sols1, label = 'Consumption')
    # # # plt.legend()
    plt.show()

# print(timeit(ga_optimization, number=1))




#######
# PLOT DE FITNESS VS ALPHA
#######

# alphas = np.linspace(0,1,21)
# store = [0.5174, 0.46, 0.41, 0.352, 0.2978, 0.2445, 0.1875, 0.1353, 0.077, 0.0418, 0.02449, 0.02429, 0.024348, 0.033, 0.037, 0.04175, 0.045, 0.046, 0.051, 0.057, 0.059]
# speed = [0.38, 0.3956, 0.4037, 0.387, 0.3985, 0.4058, 0.4109, 0.395, 0.41, 0.6725, 0.9288, 0.9676, 0.9717, 0.99956, 1.007, 1.01, 1.01, 1.02, 1.035, 1.02, 1.018]
# cons = [0.45, 0.4611, 0.467, 0.457, 0.4587, 0.4652, 0.4734,0.46, 0.47, 0.6655, 0.8966, 0.912036, 0.9173, 0.935, 0.9498, 0.9576, 0.9685, 0.9631, 0.972, 0.97, 0.9673]
#
# plt.figure(1)
# plt.plot(alphas, store)
# # ax = plt.axes(projection='3d')
# # ax.plot_surface(alpha, betha, store)
# plt.xlabel(r'$\alpha$')
# plt.ylabel('Fitness')
#
# plt.figure(2)
# plt.plot(alphas, store, label='Fitness')
# plt.plot(alphas, speed, label='V/V*')
# plt.plot(alphas, cons, label='C/C*')
# plt.xlabel(r'$\alpha$')
# plt.legend()
# # plt.ylabel('Fitness')
#
# fig, ax1 = plt.subplots()
# ax1.plot(alphas, store, label='Fitness')
# ax1.set_xlabel(r'$\alpha$')
# # ax1.tick_params('x', labelrotation=90)
# ax1.set_ylabel('Fitness')
# ax2 = ax1.twinx()  # Define a new ax with the same x axis
# ax2.set_ylabel('V/V*, C/C*')
# ax2.plot(alphas, speed, label='V/V*', color='red')
# ax2.plot(alphas, cons, label='C/C*', color='green')
# fig.legend()
# fig.tight_layout()  # Otherwise, the y axis will be a little displaced
# plt.show()