import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter


types = {'Selection': 'sel', 'Crossover': 'cross', 'Mutation': 'mut'}
functions_to_names = {'Selection': {'sel_best': 'Best',
                                    'sel_tournament': 'Tournament',
                                    'sel_roulette': 'Roulette Wheel',
                                    'sel_lexicase': 'LexiCase',
                                    'sel_lexicase_modified': 'Modified LexiCase'},
                      'Crossover': {'cross_one_point': 'One Point',
                                    'cross_two_points': 'Two Points',
                                    'cross_uniform': 'Uniform',
                                    'cross_one_child': 'One Point Unitary Child',
                                    'cross_ordered': 'Ordered (OX1)',
                                    'cross_partially_matched': 'Partially Matched (PMX)',
                                    'cross_blend': 'Blend',
                                    'cross_blend_modified': 'Modified Blend'},
                      'Mutation': {'mut_random_one_bit': 'One Bit',
                                   'mut_random_n_bit': 'N Bit',
                                   'mut_gaussian': 'Gaussian',
                                   'mut_flip_bit': 'Flip Bit',
                                   'mut_shuffle_bits': 'Shuffle Bits'}}

input_to_types = {'0': ['Selection',
                          'Selection Operators',
                          'Best fitness for different selection algorithms'],
                      '1': ['Crossover',
                          'Crossover Operators',
                          'Best fitness for different crossover algorithms'],
                      '2': ['Mutation',
                          'Mutation Operators',
                          'Best fitness for different mutation algorithms']}


if __name__ == '__main__':
    path = '../dat/saved_for_plotting/'
    files = os.listdir(path)
    print(files)

    print('Seleccione el operador a analizar: | 0 --> Selection | 1 --> Crossover | 2 --> Mutation |')
    g = input()

    type = input_to_types[g][0]
    head = types[type]

    plt.figure(1)
    n = 0
    for file in files:
        if file.split('_')[0] == head:
            n += 1

            data = np.load(path + file)

            x = np.arange(len(data[0])) + 1
            fit = data[0, :]
            std = data[1, :]

            y = savgol_filter(fit, 15, 1)
            y1 = savgol_filter(fit + std, 15, 3)
            y2 = savgol_filter(fit - std, 15, 3)

            lab = functions_to_names[type][file.split('.')[0]]
            # plt.plot(x, y1, x, y2, alpha=0.3)
            plt.plot(x, y, alpha=0.7,  label=lab)
            # plt.fill_between(x,y1,y2, alpha=0.2, label=lab)

    plt.title(input_to_types[g][2])
    plt.xlabel('# generations')
    plt.ylabel('Fitness')
    plt.legend()
    # plt.xticks(range(0,60,10), range(0,120,20))
    plt.show()


if __name__ == '__main13__':
    path = '../dat/saved_for_plotting/n_parents/'
    files = os.listdir(path)
    max_vals = []
    for file in files:
        data = np.load(path + file)
        plt.plot(data)

        x = np.arange(len(data[0])) + 1
        fit = data[0, :]
        std = data[1, :]

        lab = file.split('.')[0][1:]
        
        y = savgol_filter(fit, 11, 3)
        y1 = savgol_filter(fit + std, 11, 3)
        y2 = savgol_filter(fit - std, 11, 3)
        max_vals.append(max(y))
        # lab = functions_to_names[type][file.split('.')[0]]
        # plt.plot(x, y1, x, y2, alpha=0.3)
        plt.plot(x, y, alpha=1, label='n = {}'.format(lab))
        # plt.fill_between(x,y1,y2, alpha=0.2, label=lab)

    # plt.title(input_to_types[g][2])
    plt.legend()
    plt.show()
    #
    plt.plot(range(5,50,5), max_vals)
    plt.show()

