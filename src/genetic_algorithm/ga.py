import numpy as np
from functools import partial

# import tools




def _selection(function, population, *args, **kwargs):

    # parents = select.function(population, *args, **kwargs)
    # pfunc = partial(function, *args, **kwargs)
    # pfunc.__name__ = alias
    # pfunc.__doc__ = function.__doc__
    selection(function, population, *args, **kwargs)
    # my_sel(function, population, *args, **kwargs)



#TODO change this. I took size=len(population), but the argument population should be the parents and size should
# correspond to the children's size, which means: size = len(population) - len(psrents)
def crossover(function, population, *args, **kwargs):

    size = len(population)
    new_ind = []
    for i in range(size):
        individual_1 = population[i]
        individual_2 = population[(i + 1) % size]
        new_ind.append(_crossover.function(individual_1, individual_2, *args, **kwargs))

    return new_ind

#TODO change this. Maybe to map a function with args and kwargs is not that easy. May map-reduce help?
def mutation(function, population, *args, **kwargs):
    fun = mutation_.function
    new_population = [_mutation.function(individual, *args, **kwargs)
                      for i, individual in enumerate(population)]

    return new_population


if __name__ == '__main__':
    _selection('selTournament', 10)
    print(selection.selTournament.__doc__)