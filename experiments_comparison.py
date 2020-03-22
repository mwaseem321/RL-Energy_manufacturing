#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 15:18:07 2020

@author: Wenqing Hu (Missouri S&T)

Title: Experiment for the paper <<Novel Deep Reinforcement Learning Algorithms applied to Joint Control 
                                    of Manufacturing andOnsite Microgrid System>>

Experiment consists of 
1. Comparison of the total cost, total throughput and total energy demand for the 
optimal policy selected by reinforcement learning and the random policy; 
2. Comparison of the total cost, total throughput and total energy demand for the 
optimal policy selected by reinforcement learning and the routine straregy via mixed-integer programming.
"""

from microgrid_manufacturing_system import Microgrid, ManufacturingSystem
from reinforcement_learning import Reinforcement_Learning_Training, Reinforcement_Learning_Testing, Benchmark_RandomAction_Testing
from Simple_Manufacturing_System_routine_strategy import Mixed_Integer_Program

import numpy as np
import matplotlib.pyplot as plt


#set the number of machines
number_machines=5
#set the unit reward of production
unit_reward_production=5/100


import pandas as pd
#read the solar irradiance and wind speed data from file#
file_SolarIrradiance = "SolarIrradiance.csv"
file_WindSpeed = "WindSpeed.csv"
file_rateConsumptionCharge = "rate_consumption_charge.csv"

data_solar = pd.read_csv(file_SolarIrradiance)
solarirradiance = np.array(data_solar.iloc[:,3])

data_wind = pd.read_csv(file_WindSpeed)
windspeed = np.array(data_wind.iloc[:,3])/1000

data_rate_consumption_charge = pd.read_csv(file_rateConsumptionCharge)
rate_consumption_charge = np.array(data_rate_consumption_charge.iloc[:,4])/10


#the initial learning rates for the theta and omega iterations#
lr_theta_initial=0.003
lr_omega_initial=0.0003

#the discount factor gamma when calculating the total cost#
gamma=0.999

#number of training and testing iterations#
training_number_iteration=100
testing_number_iteration=100




"""
Testing for the Routine Strategy Selected by Mixed Integer Programming
at Given Horizon
"""
def RoutineStrategy_Testing(number_iteration, #the number of testing iterations
                            target_output     #the target output
                            ):
    
    #open and output the results to the file routine_output.txt
    rtoutput = open('routine_output.txt', 'w')

    #Calculate and output the total cost, total throughput and total energy demand for mixed-integer programming with target output as the one given by the optimal strategy
    print("\n************************* Mixed Integer Programming with given Target Output *************************", file=rtoutput)
    print("***Run the system on routine policy by mixed-integer programming at a time horizon=", number_iteration,"***", file=rtoutput)
    target_output=int(target_output)
    print("Target Output =", target_output, file=rtoutput)
    routine_sol=Mixed_Integer_Program(target_output)
    print("Optimal solution from mixed-integer programming is given by \n", routine_sol.T, file=rtoutput)

    #close and save the results to the file
    rtoutput.close()

    return 0





"""
Reinforcement Learning Algorithm: Off policy TD control combined with actor-critique
Algorithm 1 in the paper

When optimal policy is found, must add
1. Total cost and throughput in given time horizon that the 
    algorithm is used to guide the bilateral control.
2. Total energy demand across all time periods of the given 
    time horizon and the proportion of the energy supply to satisfy the demand. 
   
Two comparisons are made at the test: 
1. Comparison of the total cost, total throughput and total energy demand for the 
    optimal policy selected by reinforcement learning and the random policy; 
2. Comparison of the total cost, total throughput and total energy demand for the 
    optimal policy selected by reinforcement learning and the routine straregy via mixed-integer programming.
"""

if __name__ == "__main__":
    #initialize the grid and the manufacturing system
    grid=Microgrid(workingstatus=[0,0,0],
                   SOC=0,
                   actions_adjustingstatus=[0,0,0],
                   actions_solar=[0,0,0],
                   actions_wind=[0,0,0],
                   actions_generator=[0,0,0],
                   actions_purchased=[0,0],
                   actions_discharged=0,
                   solarirradiance=0,
                   windspeed=0
                   )
    System=ManufacturingSystem(machine_states=["Opr" for _ in range(number_machines)],
                               machine_control_actions=["K" for _ in range(number_machines)],
                               buffer_states=[2 for _ in range(number_machines-1)],
                               grid=grid
                               )
    
    #randomly generate an initial theta and plot the bounday of the simplex where theta moves#
    r=np.random.uniform(0,1,size=6)
    
    #initialize the theta variable#
    theta=[r[0]*r[1], r[0]*(1-r[1]), r[2]*r[3], r[2]*(1-r[3]), r[4]*r[5], r[4]*(1-r[5])] 
    #record the initial theta applied before training
    thetainit=theta
    
    x = [[0, 0], [0, 1], [1, 0]] 
    y = [[0, 1], [1, 0], [0, 0]]
    plt.figure(figsize = (14,10))
    for i in range(len(x)): 
        plt.plot(x[i], y[i], color='g')
    
    
    theta, omega, my_critic = Reinforcement_Learning_Training(System, 
                                                              thetainit, 
                                                              lr_theta_initial, 
                                                              lr_omega_initial, 
                                                              training_number_iteration)
    
    
    #with the optimal theta and optimal omega at hand, run the system at a certain time horizon#
    #output the optimal theta and optimal omega#
    thetaoptimal=theta
    omegaoptimal=omega  
    my_critic_optimal=my_critic

    #initialize the grid and the manufacturing system
    grid=Microgrid(workingstatus=[0,0,0],
                   SOC=0,
                   actions_adjustingstatus=[0,0,0],
                   actions_solar=[0,0,0],
                   actions_wind=[0,0,0],
                   actions_generator=[0,0,0],
                   actions_purchased=[0,0],
                   actions_discharged=0,
                   solarirradiance=0,
                   windspeed=0
                   )
    System=ManufacturingSystem(machine_states=["Opr" for _ in range(number_machines)],
                               machine_control_actions=["K" for _ in range(number_machines)],
                               buffer_states=[2 for _ in range(number_machines-1)],
                               grid=grid
                               )

    totalcostlist_optimal, totalthroughputlist_optimal, totalenergydemandlist_optimal, RL_target_output = Reinforcement_Learning_Testing(System, 
                                                                                                                                         thetainit, 
                                                                                                                                         thetaoptimal, 
                                                                                                                                         omegaoptimal, 
                                                                                                                                         my_critic_optimal, 
                                                                                                                                         testing_number_iteration, 
                                                                                                                                         unit_reward_production)
    
    #As benchmark, with initial theta and randomly simulated actions, run the system at a certain time horizon#
    
    #initialize the grid and the manufacturing system
    grid=Microgrid(workingstatus=[0,0,0],
                   SOC=0,
                   actions_adjustingstatus=[0,0,0],
                   actions_solar=[0,0,0],
                   actions_wind=[0,0,0],
                   actions_generator=[0,0,0],
                   actions_purchased=[0,0],
                   actions_discharged=0,
                   solarirradiance=0,
                   windspeed=0
                   )
    System=ManufacturingSystem(machine_states=["Opr" for _ in range(number_machines)],
                               machine_control_actions=["K" for _ in range(number_machines)],
                               buffer_states=[2 for _ in range(number_machines-1)],
                               grid=grid
                               )

    totalcostlist_benchmark, totalthroughputlist_benchmark, totalenergydemandlist_benchmark, random_target_output = Benchmark_RandomAction_Testing(System, 
                                                                                                                                                   thetainit, 
                                                                                                                                                   testing_number_iteration, 
                                                                                                                                                   unit_reward_production 
                                                                                                                                                   )

    #plot and compare the total cost, the total throughput and the total energy demand for optimal control and random control (benchmark)#
    #plot the total cost#
    plt.figure(figsize = (14,10))
    plt.plot(totalcostlist_optimal, '-', color='r')
    plt.plot(totalcostlist_benchmark, '--', color='b')
    plt.xlabel('iteration')
    plt.ylabel('total cost')
    plt.title('Total cost under optimal policy (red, solid) and benchmark random policy (blue, dashed)')
    plt.savefig('totalcost.png')
    plt.show()  

    #plot the total throughput#
    plt.figure(figsize = (14,10))
    plt.plot(totalthroughputlist_optimal, '-', color='r')
    plt.plot(totalthroughputlist_benchmark, '--', color='b')
    plt.xlabel('iteration')
    plt.ylabel('total throughput')
    plt.title('Total throughput under optimal policy (red, solid) and benchmark random policy (blue, dashed)')
    plt.savefig('totalthroughput.png')
    plt.show()  

    #plot the total energy demand#
    plt.figure(figsize = (14,10))
    plt.plot(totalenergydemandlist_optimal, '-', color='r')
    plt.plot(totalenergydemandlist_benchmark, '--', color='b')
    plt.xlabel('iteration')
    plt.ylabel('total energy demand')
    plt.title('Total energy demand under optimal policy (red, solid) and benchmark random policy (blue, dashed)')
    plt.savefig('totalenergydemand.png')
    plt.show()  
    

    """
    The 2nd Comparision Test: Comparison of the total cost, total throughput and total energy demand for the 
        optimal policy selected by reinforcement learning and the routine strategy selected by the mixed-integer programming;        
    """
    target_output=int(RL_target_output)
    
    RoutineStrategy_Testing(testing_number_iteration, target_output)
