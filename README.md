In this project, we try to proof that it is possible to reduce pollution and increase average speed of the cars in traffic jams on big roundabout streets. 

For that purpose, the NaSch model is being used for modelling the behaviour of traffic in a discrete environment. The optimization problem is solved using genetic algorithms, that will calculate the best combination of traffic lights times to increase average speed and reduce average fuel consumption.

The file "nasch.py" generates all the necessary functions for running the NaSch model. The original model is running on a roundabout, but we have also modified it to run on an "open street". We have also included entry and exit points to the roads to simulate incoming and outgoind traffic flow, as well as traffic lights in order to control the traffic state.

The file "optimization.py" serves as a main file where the input parameters are given in order to generate the problem. Also, the genetic algorithm is run in order to try to optimize the traffic behaviour during traffic jams.

