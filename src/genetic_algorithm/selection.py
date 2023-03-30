import random
import numpy as np
import copy


class selection:

    def __init__(self):
        '''Binds the name of the function given as an argument to its real
        function.'''

        # First, the dictionary of functions is defined
        self.sel_functions = {'sel_random': self.sel_random,
                              'sel_best': self.sel_best,
                              'sel_tournament': self.sel_tournament,
                              'sel_roulette': self.sel_roulette,
                              'sel_lexicase': self.sel_lexicase}

    def __call__(self, function, population, *args, **kwargs):
        # A function object is created from the input function
        self.func = self.sel_functions[function]
        # The function is called with its corresponding arguments
        return self.func(population, *args, **kwargs)

    def sel_random(self, population, n):
        """
        Select n individuals randomly chosen from the input population

        :param population: list of individuals from which the selection will be
         done.
        :param n: The number of individuals to select.
        :returns: A list containing the selected individuals.

        This function uses the :func:`~random.choice` function from the
        python base :mod:`random` module.
        """
        return random.sample(population, n)

    def sel_best(self, population, n, fitness):
        """
        Select the best n individuals from the input population

        :param population: list of individuals from which the selection will be
         done.
        :param n: The number of individuals to select.
        :param fitness: Selection criteria.
        :returns: A list containing the selected individuals.

        This function uses the :func:`~random.choice` function from the
        python base :mod:`random` module.
        """
        sorted_pop = [ind for fit, ind in sorted(zip(fitness, population))][::-1]
        selected = sorted_pop[:n]

        return selected

    def sel_tournament(self, population, n, fit, tournament_size):
        """Select the best individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.

        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        # First, the individuals in population and their fitnesses are zipped. Some
        #  of them are randomly selected and the best one is chosen. winner[0] stands
        #  for the individual itself while winner[1] would be its fitness
        fitness = copy.deepcopy(fit)
        selected = []
        for _ in range(n):
            players = random.sample(list(zip(fitness, population)), tournament_size)
            winner = next(ind for (fit, ind) in players if fit == max(
                fit for (fit, ind) in players))
            selected.append(winner)
            # If no individuals should be repeated, winner's fitness must be
            # reinitialized
            fitness[population.index(winner)] = -99

        return selected

    def sel_roulette(self, population, n, fitness):
        """Select *k* individuals from the input *individuals* using *k*
        spins of a roulette. The selection is made by looking only at the first
        objective of each individual. The list returned contains references to
        the input *individuals*.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.

        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.

        .. warning::
           The roulette selection by definition cannot be used for minimization
           or when the fitness can be smaller or equal to 0.
        """

        # Fitness and population are zipped together and sorted by fitness.
        #  Then, values are unzipped. This operation returns tuples, so they
        #  are converted into lists
        sorted_pop, sorted_fit = zip(*[(ind, fit) for fit, ind in
                                       sorted(zip(fitness, population))])
        sorted_pop, sorted_fit = list(sorted_pop), list(sorted_fit)
        total_fit = sum(fitness)
        selected = []
        for _ in range(n):
            spin_value = random.random() * total_fit
            value = 0
            for j, ind in enumerate(sorted_pop):
                value += sorted_fit[j]
                if value > spin_value:
                    selected.append(ind)

                    break

        return selected

    def sel_lexicase(self, population, n, fitness, weights):
        """Returns an individual that does the best on the fitness cases when
        considered one at a time in random order.
        http://faculty.hampshire.edu/lspector/pubs/lexicase-IEEE-TEC.pdf

        :param population: A list of individuals to select from.
        :param n: The number of individuals to select.
        :returns: A list of selected individuals.
        """
        selected = []
        used_fitness = copy.copy(fitness)
        for _ in range(n):

            candidates = list(zip(population, used_fitness))
            cases = list(range(len(fitness[0])))
            random.shuffle(cases)

            while len(cases) > 0 and len(candidates) > 1:

                cand_fitness = [fit for ind, fit in candidates]
                if weights[cases[0]] > 0:
                    best_in_case = max([x[cases[0]] for x in cand_fitness])
                else:
                    best_in_case = min([x[cases[0]] for x in cand_fitness])

                candidates = [(ind, fit) for ind, fit in candidates
                              if fit[cases[0]] == best_in_case]
                cases.pop(0)
            winner_id = random.choice(range(len(candidates)))
            selected.append(candidates[winner_id][0])
            used_fitness[population.index(candidates[winner_id][0])] = \
                [-99 for i in weights]
        return selected

__all__ = ['selection']