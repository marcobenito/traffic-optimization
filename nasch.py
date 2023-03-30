# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:37:56 2019

@author: marco
"""

"""
The nasch library contains several functions neccessary for running the 
program, which simulates the traffic using the Nagel & Schreckenberg model.
"""



import numpy as np
from numba import njit, int32



@njit(cache=True)
def probability(r):
    """ a = 1 with a probability of r
        a = 0 with a probability of 1 - r"""
    
    p=np.random.uniform(0,1)
    a = 1 if p<r else 0
    
    return(a)
    
@njit(cache=True)    #In this function, numba slows down instead
def initial_scenario(t, n, density):
    """A numpy array structure is created for simulating the road. It's filled with 0
    and the first row (the road at the beginning) is filled with cars with no speed (v = 0).
    """
    #starting with stopped cars worked faster and gave really similar results to giving different speeds to each car

    A = np.zeros((t,n), dtype=np.int64) #Array where all the info about the road (speeds at each position) is to be stored
    for i in range(n):
        A[0,i] = probability(density) - 1
    return A


#------------------------------------------------------------------------------
"""                         FUEL CONSUMPTION                                """
#------------------------------------------------------------------------------
@njit(cache=True)
def fuel_consumption(u1,u2,r,car_pos, ITL):
    """The fuel consumption is calculated by giving defined values to each speed
    :parameter: u1 is the speed a car has in the i-th time iteration
    :parameter: u2 is the speed a car will have in the (i+1)-th iteration
    :parameter: r is the random braking probability
    :return: l_h is the result of litres per hour consumed by the car
    :return: l_km is the result of litres per 100 km consumed by the car
    """
    #TODO improve the fuel consumption function. At least, take the same I had on previous versions of the code
    # It would be better if we create a real function where, if more than 5 speeds are computed, an easy interpolation
    # would give us the desired value

    # acel is computing a car's acceleration. It means: if a car has increased speed, an extra amount of fuel has been
    # consumed during the acceleration.
    # akm is the same but for l_km
    (acel, akm) = (0.8, 10) if u2 >= u1 + 1 else (0, 0)
#
    #If the car hasn't randomly braked, and it accelerated or kept velocity, the consumption is computed, whereas if
    # the car has braked, the consumption is set to 0
    # if r == 0 and u2 >= u1:
    if u2 >= u1:
    # if r==0:
        if u2 == 0:
            # If the car is stopped, it must be checked whether it's due to a
            # traffic light being red or to a traffic jam.
            n_tl = ITL.shape[1] #Number of Traffic lights in the road
            # The TL in front of the car must be found
            next_tl_id = -1
            for i in range(n_tl):
                if ITL[0,i] >= car_pos:
                    next_tl_id = i
                    break
            # If next_tl_id hasn't been declared, it means the car stands between
            # the end of the roundabout and the first TL.
            # TODO what would happen if the road is not a roundabout?
            if next_tl_id == -1:
                next_tl_id = 0

            # If the TL is red, maybe the driver will shut the motor off
            if ITL[1, next_tl_id] == 0:
                # With a probability of 70%, the driver will stop the motor
                motor_shutdown = probability(0.7)
                l_h = (0.78 * (1+acel)) * (1 - motor_shutdown)
                l_km = (30 + akm) * (1 - motor_shutdown)
            # Otherwise, the car will be stopped but not because of the TL
            else:
                l_h = 0.78 * (1+acel)
                l_km = 30 + akm
        # if u2 == 0:
        #     l_h = 0.78 * (1 + acel)
        #     l_km = 30 + akm


        elif u2 == 1:
            l_h = 2.25*(1+acel)
            l_km = 15 + akm
        elif u2 == 2:
            l_h = 3*(1+acel)
            l_km = 10 + akm
        elif u2 == 3:
            l_h = 3.6*(1+acel)
            l_km = 8 + akm
        elif u2 == 4:
            l_h = 4.8*(1+acel)
            l_km = 8 + akm
        else:
            l_h = 6*(1+acel)
            l_km = 8 + akm
            
    else:
        l_h = l_km = 0
            
#IMPLEMENTACION DE ESTO CON DICCIONARIOS ES MAS LENTOS Y CON NUMBA ES UN HORROR

    # The next dictionary has the following structure: keys are the velocities,
    # and values are a list in which the first index refers to the mean liters
    # per hour consumed by a car travelling at that speed, and the second index
    # refers to the litres consumed every 100km (that´s a measure more familiar
    # to us)
#    consumption = {0:np.asarray([0.78, 30]), 1:np.asarray([2.25,15]), \
#                   2:np.asarray([3,10]), 3:np.asarray([3.6,8]), \
#                   4:np.asarray([4.8,8]), 5:np.asarray([6,8])}
#    
#    if r == 0 and u2 >= u1:
#        
#        l_h = consumption[u2][0]*(1+acel)
#        l_km = consumption[u2][1] + akm
#    else:
#        l_h = l_km = 0
#        
    return l_h, l_km            
           
            
#------------------------------------------------------------------------------
"""                           TRAFFIC LIGHTS                                """
#------------------------------------------------------------------------------
@njit(cache=True)
def initialize_trafficlight(ntl,n):
    """Information about traffic lights is given in a numpy array called ITL
    (Intelligent Traffic Light). There is one column for each traffic light,
    and the rows correspond to: 
        First row: Position of the TL
        Second row: Phase (0 for red, 1 for green)
        Third row: Time in current phase
        Fourth row: Total time for red phase
        Fifth row: total time for green phase"""
    ITL = np.zeros((5, ntl), dtype=np.int64)
    for i in range(ntl):
        ITL[0,i] = int( (i+1) * ((n-1)/ntl))
        
    # ITL[0,:] = [50,150]
    
    #All traffic lights are initialized to green    
    ITL[1,:] = 1
    ITL[3:,:]= 1E10

    return ITL

@njit(cache=True)
def settrafficlight(ITL, x):
    """Traffic lights are set to green or red when it corresponds
    :parameter: ITL is the traffic light structure defined in the function initialize_trafficlight
    :parameter: x is the vector showing the current state of the road"""
    for i in range(ITL.shape[1]):
        #if TL in green, it's cheked whether the time in current phase is
        #greater than the total time for the phase or not. If it's bigger, 
        #phase is changed to red and time in current phase reinitialized to 1.
        if ITL[1,i] == 1:
            phase_time = ITL[4,i]
            if ITL[2,i] <= phase_time:
                ITL[2,i] += 1
            elif ITL[2,i] > phase_time:
                ITL[1,i] = 0
                ITL[2,i] = 1
        #Same but if TL in red
        elif ITL[1,i] == 0:
            phase_time = ITL[3,i]
            if ITL[2,i] <= phase_time:
                ITL[2,i] += 1
            elif ITL[2,i] > phase_time:
                ITL[1,i] = 1
                ITL[2,i] = 1
        #Finally, if the TL is in red, the speed value of the position where 
        #the TL is located is set to -2.        
        if ITL[1,i] == 0:
            if x[ITL[0,i]] == -1:
                x[ITL[0,i]] = -2
    
    return ITL, x


@njit(cache=True)
def entry(u,c, density):
    """A car will enter the road with a certain probability. This probability is set to be the same as de density of
    cars in the road, so that the procedure of entrance is smooth.
    :parameter: u is the information in the first space of the road.
    :parameter: c is a number that sets the accumulated cars that want to enter the road. It's updated in the function.
    :parameter: density is the density of cars in the road.

    :return: u is updated to the speed of the incoming car, if there is one.
    :return: c"""


    if u > -1:
        u = u
        # c = c + probability(density)
    # if there is an empty position at the beggining of the road, first it's checked whether there are accumulated cars
    # asking for entrance or not.
    elif u  == -1:
        pp = probability(density)
        u = -1 + 2*pp
        c += pp
        # if c > 0:
        #     u = 1       #In case there are accumulated cars, one of them enters the road
        #     c = c - 1   # And of course, there is one less acucumulated car
        #
        # elif c == 0:
        #     #In case there aren't accumulated cars, one could also appear and enter the road
        #     u = u + 2*probability(density)
        
    
    return u, c

@njit(cache=True)
def evaluate_exits(exit, speed):
    # print(len(exit))
    for i in range(exit.shape[1]):
        for j in range(int(exit[0,i])-3,int(exit[0,i])+2):
            if speed[j] != -1:
                pp = probability(1-exit[1,i])
                speed_i = -1 + (1 + speed[j])*pp
                if pp==0:
                    exit[2,i] += 1
                speed[j] = speed_i
                break

    return speed

@njit(cache=True)
def evaluate_entrances(entrance, speed):
    # print(len(exit))
    # print('entre')
    n = len(speed)
    for i in range(entrance.shape[1]):
        position = int(entrance[0, i])
        entrance_probability = entrance[1, i]

        if speed[position] > -1:
            entrance[2, i] += 0#probability(entrance_probability) #entrance[2, i] is the accumulated cars counter

        elif speed[position] == -1:
            distance_to_previous_car = 1
            distance_to_next_car = 1
            while position + distance_to_next_car < n:
                if speed[position + distance_to_next_car] == -1:
                    distance_to_next_car += 1
                else:
                    break
            while speed[(position - distance_to_previous_car)] == -1:
                distance_to_previous_car += 1

            speed_of_previous_car = speed[position - distance_to_previous_car]
            # speed_of_next_car = speed[position + distance_to_next_car]
            speed_of_new_car = -1

            # if entrance[2, i] > 0:

            if distance_to_previous_car >= speed_of_previous_car - 2 and (distance_to_next_car - 1) > 0:
                pp = probability(entrance_probability)
                speed_of_new_car = min(-1 + (distance_to_next_car) * pp, 5)
                entrance[2,i] += pp


                    # entrance[2, i] -= 1

            # elif entrance[2, i] == 0:
            #     speed_of_new_car = (distance_to_next_car - 1) * probability(entrance_probability)

            speed[position] = speed_of_new_car
    # print('c = ', entrance[2,:])
    # print('sali')
    return entrance, speed


@njit(cache=True)
def roundabout(U, P, P0, vmax, ITL,entrance=None, exits=None):
    """This is the general NaSch code

    :parameter: U is the array where the information of the road is stored. Each row stands for a time iteration. Each
    column stands for a car's position. The values are the speeds of the cars. A value of -1 indicates an empty space,
    and a value of -2 indicates an empty space where a traffic light is located.
    :parameter: ITL is an array containing information about the traffic lights.
    :parameter: n is the total amount of spacial positions in the road.
    :parameter: t is the total amount of time intervals .
    :parameter: P is the random brake probability.
    :parameter: vmax is the maximum speed allowed in the road.

    :return: U, after being completed, is the main return of the function.
    :return: means is an array containing the mean speed and the mean fuel consumption per hour and per 100 km."""

    n = U.shape[1]
    t = U.shape[0]
    density = 1 - abs(np.sum(U[0, :]) / n)
    v_mean = n_v = c_h_mean = c_km_mean = 0
    # print('Initial density: ', density)
    for i in range(t-1):
        ITL, U[i,:] = settrafficlight(ITL,U[i,:])
        movement = np.zeros(n,dtype=np.int64) - 1    #array with the same dimension as a row in U. Here, the positions
        # in the next time interval are calculated. It works way faster than updating the U array all the time
        if exits is not None:
            U[i,:] = evaluate_exits(exits, U[i,:])
        if entrance is not None:
            entrance, U[i,:] = evaluate_entrances(entrance, U[i,:])

        for j in range(n):
            #If there is a car in the j-th position, the function is evaluated
            if U[i,j] > -1:         #U[i,j] stores the velocity of a car in the i-th position at the j-th time step
                
                v_mean += U[i,j]    #A number for counting the speeds in the road. Used for computing the mean speed
                n_v += 1            #A number for couting the spaces occupied in the road for computing the means
                
                #Distance
                distance = 1
                while U[i,(j + distance)%n] == -1:
                    distance += 1
                #Brake probability
                p = P0 if U[i, j] == 0 else P
                random_brake = probability(p)
                #First evaluation (acceleration, distance & max speed)
                v1 = min(U[i,j]+1,distance-1,vmax)
                #Second evaluation (random brake & 0)
                vnew = max(v1-random_brake,0)
                #Fuel consumption calculation
                c1, c2 = fuel_consumption(U[i,j], vnew,random_brake, j, ITL)
                c_h_mean += c1
                c_km_mean += c2
                #Movement
                movement[(j + vnew) - n] = vnew
#                
        U[i+1,:] = movement
        
    v_mean /= n_v
    c_h_mean /= n_v
    c_km_mean /= n_v
    
    means = (v_mean, c_h_mean, c_km_mean)

    return U, means

@njit(cache=True)
def straightroad(U, ITL, n, t, density, P, P0, vmax, entrance=None, exits=None, flag_entry=False):
    #
    # P0 = 0.5
    v_mean, n_v, c_h_mean, c_km_mean = 0, 0, 0, 0
    counter = 0
    
    for i in range(t-1):

        ITL, U[i,:] = settrafficlight(ITL,U[i,:])
        mov = np.zeros(n,dtype=np.int64) - 1
        #Incoming cars
        if flag_entry:
            U[i,0], counter = entry(U[i,0], counter, density)
        if exits is not None:
            U[i,:] = evaluate_exits(exits, U[i,:])
        # if entrance is not None:
        #     entrance, U[i,:] = evaluate_entrances(entrance, U[i,:])

        for j in range(n):      
            if U[i,j] > -1:   
                
                v_mean += U[i,j]
                n_v += 1
                
                #Distance
                d = 1
                while j+d < n:
                    if U[i,(j + d)] == -1:
                        d += 1
                    else:
                        break
                if (j+d) >= n:
                    d = 5


                #Brake probability
                p = P0 if U[i,j] == 0 else P
                random_brake = probability(p)
                #First evaluation (acceleration, distance & max speed)
                v1 = min(U[i,j]+1,d-1,vmax)
                #Second evaluation (random brake & 0)
                vnew = max(v1-random_brake,0)
                #Fuel consumption calculation
                # c1, c2 = fuel_consumption(U[i,j], vnew, j, ITL)
                c1, c2 = fuel_consumption(U[i, j], vnew, random_brake, j, ITL)
                c_h_mean += c1
                c_km_mean += c2
                #Movement
                if (j + vnew) < n:
                    mov[j + vnew] = vnew
#                
        U[i+1,:] = mov 
        
    v_mean /= n_v
    c_h_mean /= n_v
    c_km_mean /= n_v
    # print('Total amount of cars tha entered the road: ', counter)
    means = (v_mean, c_h_mean, c_km_mean)
    
    return U, means
