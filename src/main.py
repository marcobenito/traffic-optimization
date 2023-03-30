# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:15:20 2019

@author: marco
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import nasch
from timeit import timeit


n = 150            #number of space positions
t = 500           #number of space intervals
density = 0.7     #density of cars in the road
P = 0.3             #Random brake probability
P0 = 0.3            #Random brake probability for stopped cars
vmax = 5            #Maximum road speed
ntl = 2            #number of traffic lights in the road


# The entries and exits arrays must be defined. They are as follows:
#     First row: position of the entry/exit
#     Second row: Probability that a car enters os exits through this entry/exit
#     Third row: Only applies for entries. Number of cars accumulated waiting for entering
entrance_positions = np.array([120, 540],dtype=np.int64)
entrance_probability = np.array([0.15, 0.07],dtype=np.int64)
exit_positions = np.array([50, 210, 430],dtype=np.int64)
exit_probability = np.array([0.08, 0.1, 0.07],dtype=np.int64)
entrance_positions = np.array([120, 540],dtype=np.int64)
entrance_probability = np.array([0, 0],dtype=np.int64)
exit_positions = np.array([50, 210, 430],dtype=np.int64)
exit_probability = np.array([0, 0, 0],dtype=np.int64)

entrance = np.vstack((entrance_positions, entrance_probability, np.zeros(entrance_positions.size)))
exit = np.vstack((exit_positions, exit_probability, np.zeros(exit_positions.size)))


def call(fun, *args, **kwargs):
    return fun(*args, *kwargs)

it = time()
# @njit
def main(P, P0, density):
#First, the road and the traffic lights are initialized
    A = nasch.initial_scenario(t, n, density)
    ITL = nasch.initialize_trafficlight(ntl,n)
    # ITL[1, :] = 0
    ITL[3,:] = [500,500]#[73,96,85]#[118,112,115]
    ITL[4,:] = [500,500]#[379,378,285]#[310,320,316]#
#Next, the space-time diagram and fuel consumption are obtained
    # arguments = (A, ITL,entrance, exit, P, vmax)

    # A, means = call(nasch.roundabout, A, P, vmax, ITL,entrance, exit)
    # A, means = nasch.straightroad(A, ITL, n, t, density, P, vmax, flag_entry = True)
    # A, means = call(nasch.straightroad, A, ITL, n, t, density, P, P0, vmax, flag_entry = True)
    A, means = nasch.straightroad(A, ITL, n, t, density, P, P0, vmax, flag_entry = True)
    return A, means

A, means = main(P, P, density)
print(means)
v_mean = means[0]
c_mean = means[1]

ft = time()
print(ft-it)
# print(timeit(main,number = 1000))



for i in range(t):
    for j in range(n):
        A[i,j] = False if A[i,j] <=-1 else True

x,y = np.argwhere(A == True).T
plt.figure(1, figsize=(5,7))
plt.scatter(y,x,s=0.01,c="blue")
plt.axis([0,n,t, 0])
plt.xlabel('Espacio')
plt.ylabel('Tiempo')

plt.show()
#plt.matshow(A[:,0:n])

#♥


if __name__ == '__m2ain__':
    k = 2
    pp1 = np.linspace(0,0.1,k)
    pp2 = np.array([0.15,0.3, 0.45])
    P0 = np.linspace(0,0.8,3)
    pp = np.hstack([pp1,pp2])
    density = np.linspace(0.01, 1,50)
    means_v = np.zeros((len(pp), len(P0), 50))
    means_f = np.zeros_like(means_v)
    plt.figure()
    for i, p in enumerate(pp):
        print(p)
        for ii, p0 in enumerate(P0):

            for j, dens in enumerate(density):
                mean_speed = 0
                rep = 3
                for _ in range(rep):
                    try:
                        mean_speed += main(p, p0, dens)[1][0]
                    except ZeroDivisionError:
                        pass
                mean_speed /= rep
                # print(mean_speed)
                mean_phi = mean_speed * dens
                means_v[i, ii, j] = mean_speed
                means_f[i, ii, j] = mean_phi
                # means_v[i, j] = main(p, dens)[1]

            approximate_flux = np.polyfit(density, means_f[i, ii, :], 8)
            approximate_flux = np.poly1d(approximate_flux)
            x = np.linspace(0.1, max(density), 100)
            y = approximate_flux(density)
        # plt.plot(x,y, label=day)

            plt.plot(density, means_v[i, ii, :], label='P = ' + str(p) + '; P0 = ' + str(p0))
            # plt.plot(density,y, label='P = ' + str(p) + '; P0 = ' + str(p0))
    plt.legend()
    plt.show()



if __name__ == '__main__':
    it = time()
    kk = 100
    kk1 = 10
    speeds = np.zeros((kk,kk))
    cons = np.zeros((kk,kk))
    ITL = nasch.initialize_trafficlight(ntl, n)

    for i, s1 in enumerate(np.linspace(0,t,kk)):
        print('Iteration {}'.format(i))
        for j, s2 in enumerate(np.linspace(0,t,kk)):
            speed1 = 0
            cons1 = 0
            for k in range(kk1):
                A = nasch.initial_scenario(t, n, density)
                ITL = nasch.initialize_trafficlight(ntl, n)
                ITL[1, :] = 0
                ITL[3, :] = [s1, s2]
                ITL[4, :] = 1E7
                means = nasch.straightroad(A, ITL, n, t, density, P, P, vmax, flag_entry=True)[1]
                speed1 += means[0]
                cons1 += means[1]
            speeds[i, j] = speed1/kk1
            cons[i,j] = cons1/kk1
    ft = time()
    print('Time: {}'.format(ft-it))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, t, kk)
    y = np.linspace(0, t, kk)
    X, Y = np.meshgrid(x, y)
    Z = speeds/v_mean
    Z1 = cons/c_mean

    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Time S1')
    ax.set_ylabel('Time S2')
    ax.set_zlabel('V/V*')
    ax.axis([0, t, t, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z1, cmap='viridis')

    ax.set_xlabel('Time S1')
    ax.set_ylabel('Time S2')
    ax.set_zlabel('C/C*')
    ax.axis([0, t, t, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z/Z1, cmap='viridis')

    ax.set_xlabel('Time S1')
    ax.set_ylabel('Time S2')
    ax.set_zlabel('V/V*/C/C*')
    ax.axis([0, t, t, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, (Z-1)*0.5+(1-Z1)*0.5, cmap='viridis')

    ax.set_xlabel('Time S1')
    ax.set_ylabel('Time S2')
    ax.set_zlabel('Fitness')
    ax.axis([0, t, t, 0])

    plt.show()

    plt.scatter(Z.reshape((1,kk**2)), Z1.reshape(1,kk**2))
    plt.show()



