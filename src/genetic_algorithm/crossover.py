import random
from copy import copy


class crossover:

    def __init__(self):
        '''Binds the name of the function given as an argument to its real
         function.'''

        # First, the dictionary of functions is defined
        self.cross_functions = {
            'cross_one_point': self.cross_one_point,
            'cross_two_points': self.cross_two_points,
            'cross_uniform': self.cross_uniform,
            'cross_one_child': self.cross_one_child,
            'cross_ordered': self.cross_ordered,
            'cross_partially_matched': self.cross_partially_matched,
            'cross_blend': self.cross_blend,
            'cross_blend_modified': self.cross_blend_modified
        }

    def __call__(self, function, inds_to_cross, n, cross_pb, *args, **kwargs):

        # A function object is created from the input function
        self.func = self.cross_functions[function]
        # New offsprings will be generated from the selected individuals. The
        #  amount of offsprings to generate must be n. Two new individuals are
        #  generated in each iteration, so if in one of those iterations n is
        #  reached, the while statement will break. This condition is evaluated
        #  twice in case n is even.

        offspring = []
        while True:
            ind1, ind2 = random.sample(inds_to_cross, 2)
            if random.random() < cross_pb:
                # The function is called with its corresponding arguments
                crossed_inds = self.func(copy(ind1), copy(ind2),
                                         *args, **kwargs)
                offspring.append(crossed_inds[0])
                if len(offspring) == n:
                    break
                try:
                    offspring.append(crossed_inds[1])
                    if len(offspring) == n:
                        break
                except:
                    pass

            # i = 0
            # while True:
            #     ind1, ind2 = inds_to_cross[2*i], inds_to_cross[2*i + 1]
            #     if random.random < cross_pb:
            #         crossed_inds = self.func(ind1, ind2, *args, **kwargs)
            #         offspring.append(crossed_inds[0])
            #         if len(offspring) == n:
            #             break
            #         offspring.append(crossed_inds[1])
            #         if len(offspring) == n:
            #             break


        return offspring

    def cross_one_point(self, ind1, ind2, cp=None):
        """crosse genes in one point
        :param ind1: first individual in the cross.
        :param ind2: second individual to be crossed.
        :param cp: point at which cross takes place. If not provided, it will be
        chosen randomly.
        :returns: both individuals after the cross.
        """
        size = min(len(ind1), len(ind2))
        if not cp:
            cp = random.randint(1, size - 1)
        ind1[cp:], ind2[cp:] = ind2[cp:], ind1[cp:]

        return ind1, ind2

    def cross_two_points(self, ind1, ind2, cp1=None, cp2=None):
        """Executes a two-point crossover on the input :term:`sequence`
        individuals. The two individuals are modified in place and both keep
        their original length.

        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :param cp1: first cross point (optional)
        :param cp2: second cross point (optional). If they are not give, they
        will be calculated randomly
        :returns: A tuple of two individuals.

        This function uses the :func:`~random.randint` function from the Python
        base :mod:`random` module.
        """
        size = min(len(ind1), len(ind2))
        if not cp1 and not cp2:
            cp1 = random.randint(1, size)
            cp2 = random.randint(1, size - 1)
        if cp2 >= cp1:
            cp2 += 1
        else:  # Swap the two cp points
            cp1, cp2 = cp2, cp1

        ind1[cp1:cp2], ind2[cp1:cp2] = ind2[cp1:cp2], ind1[cp1:cp2]

        return ind1, ind2

    def cross_uniform(self, ind1, ind2, pb):
        """Executes a uniform crossover that modify in place the two
        :term:`sequence` individuals. The attributes are swapped accordingto the
        *indpb* probability.

        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :param pb: Independent probabily for each attribute to be exchanged.
        :returns: A tuple of two individuals.

        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        """
        size = min(len(ind1), len(ind2))
        for i in range(size):
            if random.random() < pb:
                ind1[i], ind2[i] = ind2[i], ind1[i]

        return ind1, ind2

    def cross_one_child(self, ind1, ind2, cp=None):
        """crosse genes in one point
        :param ind1: first individual in the cross.
        :param ind2: second individual to be crossed.
        :param cp: point at which cross takes place. If not provided, it will be
        chosen randomly.
        :returns: both individuals after the cross.
        """
        size = min(len(ind1), len(ind2))
        if not cp:
            cp = random.randint(1, size - 1)
        new_ind = ind1[:cp] + ind2[cp:]

        return [new_ind]

    def cross_ordered(self, ind1, ind2):
        """Executes an ordered crossover (OX) on the input
        individuals. The two individuals are modified in place. This crossover
        expects :term:`sequence` individuals of indices, the result for any other
        type of individuals is unpredictable.

        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :returns: A tuple of two individuals.

        Moreover, this crossover generates holes in the input
        individuals. A hole is created when an attribute of an individual is
        between the two crossover points of the other individual. Then it rotates
        the element so that all holes are between the crossover points and fills
        them with the removed elements in order. For more details see
        [Goldberg1989]_.

        This function uses the :func:`~random.sample` function from the python base
        :mod:`random` module.

        .. [Goldberg1989] Goldberg. Genetic algorithms in search,
           optimization and machine learning. Addison Wesley, 1989
        """
        print(len(ind1), len(ind2))
        size = min(len(ind1), len(ind2))
        a, b = random.sample(range(size), 2)
        if a > b:
            a, b = b, a

        holes1, holes2 = [True] * size, [True] * size
        for i in range(size):
            if i < a or i > b:
                print(i)
                holes1[ind2[i]] = False
                holes2[ind1[i]] = False

        # We must keep the original values somewhere before scrambling everything
        temp1, temp2 = ind1, ind2
        k1, k2 = b + 1, b + 1
        for i in range(size):
            if not holes1[temp1[(i + b + 1) % size]]:
                ind1[k1 % size] = temp1[(i + b + 1) % size]
                k1 += 1

            if not holes2[temp2[(i + b + 1) % size]]:
                ind2[k2 % size] = temp2[(i + b + 1) % size]
                k2 += 1

        # Swap the content between a and b (included)
        for i in range(a, b + 1):
            ind1[i], ind2[i] = ind2[i], ind1[i]

        return ind1, ind2

    def cross_partially_matched(self, ind1, ind2):
        """Executes a partially matched crossover (PMX) on the input individuals.
        The two individuals are modified in place. This crossover expects
        :term:`sequence` individuals of indices, the result for any other type of
        individuals is unpredictable.

        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :returns: A tuple of two individuals.

        Moreover, this crossover generates two children by matching
        pairs of values in a certain range of the two parents and swapping the values
        of those indexes. For more details see [Goldberg1985]_.

        This function uses the :func:`~random.randint` function from the python base
        :mod:`random` module.

        .. [Goldberg1985] Goldberg and Lingel, "Alleles, loci, and the traveling
           salesman problem", 1985.
        """
        size = min(len(ind1), len(ind2))
        p1, p2 = [0] * size, [0] * size

        # Initialize the position of each indices in the individuals
        for i in range(size):
            p1[ind1[i]] = i
            p2[ind2[i]] = i
        # Choose crossover points
        cxpoint1 = random.randint(0, size)
        cxpoint2 = random.randint(0, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        # Apply crossover between cx points
        for i in range(cxpoint1, cxpoint2):
            # Keep track of the selected values
            temp1 = ind1[i]
            temp2 = ind2[i]
            # Swap the matched value
            ind1[i], ind1[p1[temp2]] = temp2, temp1
            ind2[i], ind2[p2[temp1]] = temp1, temp2
            # Position bookkeeping
            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

        return ind1, ind2

    def cross_blend_modified(self, ind1, ind2, alpha=0):
        """Executes a blend crossover that modify in-place the input individuals.
        The blend crossover expects :term:`sequence` individuals of floating point
        numbers.

        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :param alpha: Extent of the interval in which the new values can be drawn
                      for each attribute on both side of the parents' attributes.
        :returns: A tuple of two individuals.

        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        """
        for i, (x1, x2) in enumerate(zip(ind1, ind2)):
            gamma = (1. + 2. * alpha) * random.random() - alpha
            ind1[i] = int((1. - gamma) * x1 + gamma * x2)
            ind2[i] = int(gamma * x1 + (1. - gamma) * x2)

        return ind1, ind2


    def cross_blend(self, ind1, ind2):
        """Executes a blend crossover that modify in-place the input individuals.
        The blend crossover expects :term:`sequence` individuals of floating point
        numbers.

        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :param alpha: Extent of the interval in which the new values can be drawn
                      for each attribute on both side of the parents' attributes.
        :returns: A tuple of two individuals.

        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        """
        for i, (x1, x2) in enumerate(zip(ind1, ind2)):
            b = random.random()
            ind1[i] = int(x1 - b * (x1 - x2))
            ind2[i] = int(x2 + b * (x1 - x2))

        return ind1, ind2


__all__ = ['crossover']
