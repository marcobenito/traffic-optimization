import random

class mutation:
    def __init__(self):

        # First, the dictionary of functions is defined
        self.mut_functions = {'mut_random_one_bit': self.mut_random_one_bit,
                              'mut_random_n_bit': self.mut_random_n_bit,
                              'mut_gaussian': self.mut_gaussian,
                              'mut_flip_bit': self.mut_flip_bit,
                              'mut_shuffle_bits': self.mut_shuffle_bits}

    def __call__(self, function, offspring, mut_ind_pb, *args, **kwargs):

        # A function object is created from the input function
        self.func = self.mut_functions[function]

        for i, ind in enumerate(offspring):
            if random.random() < mut_ind_pb:
                offspring[i] = self.func(ind, *args, **kwargs)

        return offspring

    def mut_random_one_bit(self, individual, min_value, max_value):
        size = len(individual)
        bit = random.randint(0,size-1)
        individual[bit] = random.randint(min_value, max_value)
        return individual

    def mut_random_n_bit(self, individual, min_value, max_value, mut_bit_pb):
        """Randomly changes the values of multiple genes selected with
        a probability of mut_bit_pb

        it's the mutUniformInt function in DEAP"""

        size = len(individual)
        for bit in range(size):
            if random.random() < mut_bit_pb:
                individual[bit] = random.randint(min_value, max_value)

        return individual

    def mut_gaussian(self, individual, mu, sigma, mut_bit_pb):
        """This function applies a gaussian mutation of mean *mu* and standard
        deviation *sigma* on the input individual. This mutation expects a
        :term:`sequence` individual composed of real valued attributes.
        The *indpb* argument is the probability of each attribute to be mutated.

        :param individual: Individual to be mutated.
        :param mu: Mean or :term:`python:sequence` of means for the
                   gaussian addition mutation.
        :param sigma: Standard deviation or :term:`python:sequence` of
                      standard deviations for the gaussian addition mutation.
        :param indpb: Independent probability for each attribute to be mutated.
        :returns: A tuple of one individual.

        This function uses the :func:`~random.random` and :func:`~random.gauss`
        functions from the python base :mod:`random` module.
        """
        size = len(individual)
        # if not isinstance(mu, Sequence):
        #     mu = repeat(mu, size)
        # elif len(mu) < size:
        #     raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
        # if not isinstance(sigma, Sequence):
        #     sigma = repeat(sigma, size)
        # elif len(sigma) < size:
        #     raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

        # for i, m, s in zip(range(size), mu, sigma):
        #     if random.random() < mut_bit_pb:
        #         individual[i] += random.gauss(m, s)
        for i in range(size):
            if random.random() < mut_bit_pb:
                individual[i] += random.gauss(mu,sigma)

        return individual

    def mut_flip_bit(self, individual, mut_bit_pb):
        """Flip the value of the attributes of the input individual and return the
        mutant. The *individual* is expected to be a :term:`sequence` and the values of the
        attributes shall stay valid after the ``not`` operator is called on them.
        The *indpb* argument is the probability of each attribute to be
        flipped. This mutation is usually applied on boolean individuals.

        :param individual: Individual to be mutated.
        :param indpb: Independent probability for each attribute to be flipped.
        :returns: A tuple of one individual.

        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        """
        
        size = len(individual)
        for bit in range(size):
            if random.random() < mut_bit_pb:
                individual[bit] = type(individual[bit])(not individual[bit])

        return individual

    def mut_shuffle_bits(self, individual, mut_bit_pb):
        """Shuffle the attributes of the input individual and return the mutant.
        The *individual* is expected to be a :term:`sequence`. The *indpb* argument is the
        probability of each attribute to be moved. Usually this mutation is applied on
        vector of indices.

        :param individual: Individual to be mutated.
        :param indpb: Independent probability for each attribute to be exchanged to
                      another position.
        :returns: A tuple of one individual.

        This function uses the :func:`~random.random` and :func:`~random.randint`
        functions from the python base :mod:`random` module.
        """
        size = len(individual)
        for bit in range(size):
            if random.random() < mut_bit_pb:
                new_bit = random.randint(0, size - 2)
                if new_bit >= bit:
                    new_bit += 1
                individual[bit], individual[new_bit] = \
                    individual[new_bit], individual[bit]

        return individual

__all__ = ['mutation']