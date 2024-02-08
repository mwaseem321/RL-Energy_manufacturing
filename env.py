import random

from microgrid_manufacturing_system2 import Microgrid, ManufacturingSystem, ActionSimulation, MicrogridActionSet_Discrete_Remainder, MachineActionTree, SystemInitialize
from projectionSimplex import projection
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import pandas as pd
#read the solar irradiance and wind speed data from file#
#read the rate of consumption charge date from file#
file_SolarIrradiance = "SolarIrradiance.csv"
file_WindSpeed = "WindSpeed.csv"
file_rateConsumptionCharge = "rate_consumption_charge.csv"
#read the solar irradiace
data_solar = pd.read_csv(file_SolarIrradiance)
solarirradiance = np.array(data_solar.iloc[:,3])
#solar irradiance measured by MegaWatt/km^2
#read the windspeed
data_wind = pd.read_csv(file_WindSpeed)
windspeed = 3.6*np.array(data_wind.iloc[:,3])
#windspeed measured by km/h=1/3.6 m/s
#read the rate of consumption charge
data_rate_consumption_charge = pd.read_csv(file_rateConsumptionCharge)
rate_consumption_charge = np.array(data_rate_consumption_charge.iloc[:,4])/10
#rate of consumption charge measured by 10^4$/MegaWatt=10 $/kWh
T=100
class Env(object):
    def __init__(self):
        self.t= 0
        self.workingstatus = [0, 0, 0]
        self.actions_purchased =  [0,0]
        self.actions_discharged= 0
    def reset(self):
        # machine_obs[i]: [machine_status[i], buffer status[i-1, i]
        # solar_obs: [SOC,solar_energy]
        # wind_obs: [SOC,wind_energy]
        # generator_obs: [SOC, generator_energy]
        #[0: off, 1: opr, 2: brk, 3: sta, 4: blo]
        # return [["off", buffer levels], ["off", 0, 0], ["off", 0, 0], ["off", 0, 0], ["off", 0], [0, 0], [0, 0], [0, 0]]
        reset_list= [[0, 0,0,0, 0],[0,0,0,0,0 ],[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
        self.machine_states= [sublist[0] for sublist in reset_list[:5]]
        # print("self.machine_states: ", self.machine_states)
        self.buffer_states= reset_list[0][1:]
        # print("self.buffer_states: ", self.buffer_states)
        self.SOC= reset_list[6][0]
        return reset_list

    def step(self, actions, t):
        #actions: machine_actions: ["K,H, W, K, H"; theta: 0,1,2,3,4,5;]
        # print("actions before flattening: ", actions)
        flat_actions = []
        for action in actions:
            # print("type(action in flat actions): ", type(action))
            if isinstance(action, (list, np.ndarray)):
                flat_actions.extend(action)
            else:
                flat_actions.append(action)
        print("flat_actions in the step are: ", flat_actions)
        # theta= flat_actions[-9:-7] + flat_actions[-6:-4]+ flat_actions[-3:-1]
        # print("theta: ", theta)
        # print("actions in the step: ", flat_actions)
        # print("timestep: ", t)
        obs_= self.get_obs(flat_actions, t)
        self.machine_states = [sublist[0] for sublist in obs_[:5]]
        self.buffer_states = obs_[0][1:]
        self.SOC = obs_[6][0]
        reward= self.get_reward(t)
        done= self.is_done(t)
        info = self.get_info()
        return obs_, reward, done, info

    def get_obs(self, flat_actions, t):
        # theta = flat_actions[-9:-7] + flat_actions[-6:-4] + flat_actions[-3:-1]
        # ManufacturingSystem()

        # Random theta for testing!
        # Define the number of lists
        num_lists = 3
        # Initialize the list of lists
        list_of_lists = []

        # Generate three lists
        for _ in range(num_lists):
            # Generate two random numbers between 0 and 1
            num1 = random.random()
            num2 = random.random()

            # Ensure the third number is positive and such that the sum of all three equals 1
            # Randomize the third number based on the sum of the first two numbers
            total_sum = num1 + num2
            num3 = 1 - total_sum

            # Ensure all numbers are positive
            if num3 < 0:
                # If num3 is negative, adjust num1 or num2 accordingly
                adjustment = abs(num3)
                if num1 >= num2:
                    num1 -= adjustment
                else:
                    num2 -= adjustment

            # Create the list
            random_numbers = [num1, num2, max(0, num3)]  # Ensure num3 is positive

            # Append the list to the list of lists
            list_of_lists.append(random_numbers)

        theta = list_of_lists
        print("Random thetas generated in the step are: ", theta)
        machine_control_actions= []
        solar_actions= []
        solar_actions.extend(flat_actions[-9:-6])
        wind_actions= []
        wind_actions.extend(flat_actions[-6:-3])
        generator_actions= []
        generator_actions.extend(flat_actions[-3:])

        for i in range(0, 15, 3): # 15 represents the actions of 5 machines i.e., 5x3
            # Get the three values for each machine
            machine_actions = flat_actions[i:i + 3]

            # Determine the action based on the highest value
            highest_value = max(machine_actions)
            action_index = machine_actions.index(highest_value)

            # Assign the action based on the index
            if action_index == 0:
                action = "K"
            elif action_index == 1:
                action = "H"
            else:
                action = "W"

            # Append the action to the list of actions for the machine
            machine_control_actions.append(action)
        print("machine_control_actions in the step are: ", machine_control_actions)
        actions_adjustingstatus = [random.randint(0, 1) for _ in range(3)] #[1, 1, 1]
        print(" Random actions_adjustingstatus before initializing grid in the step are: ", actions_adjustingstatus)
        print("self.workingstatus before initializing Microgrid class: ", self.workingstatus)
        print("self.SOC before initializing Microgrid class: ", self.SOC)
        print("self.actions_purchased before initializing Microgrid class: ", self.actions_purchased)
        print("self.actions_discharged before initializing Microgrid class: ", self.actions_discharged)
        print("solar actions extracted from flat_actions before initializing Microgrid class: ", solar_actions)
        print("wind_actions extracted from flat_actions before initializing Microgrid class: ", wind_actions)
        print("generator_actions extracted from flat_actions before initializing Microgrid class: ", generator_actions)
        print("solarirradiance[t] before grid initialization: ", solarirradiance[t])
        print("windspeed[t] before grid initialization: ", windspeed[t])
        grid= Microgrid(self.workingstatus, self.SOC,actions_adjustingstatus, solar_actions,wind_actions,generator_actions,self.actions_purchased,
                        self.actions_discharged,
                                solarirradiance= solarirradiance[t], #t//8640
                                windspeed=windspeed[t], #t//8640
                                t= t)
        # Plot the updated values
        # plt.plot(grid.time_steps_plot, grid.EAT_plot, label='EAT')
        # # plt.plot(grid.time_steps_plot, grid.Dt_plot, label='Dt')
        # plt.xlabel('Time Step')
        # plt.ylabel('Value')
        # plt.title('Real-time Plot')
        # plt.legend()
        # plt.pause(0.05)  # Pause for a short duration to update the plot
        # plt.clf()  # Clear the current figure
        # plt.show()
        print("self.machine_states before initializing manufacturingSystem class: ", self.machine_states)
        print("self.buffer_states before initializing manufacturingSystem class: ", self.buffer_states)
        print("machine_control_actions before initializing manufacturingSystem class: ", machine_control_actions)
        self.system= ManufacturingSystem(self.machine_states, machine_control_actions, self.buffer_states, grid)
        print("self.machine_states before transition: ", self.machine_states)
        print("Buffer states before transition: ", self.buffer_states )
        self.machine_states, self.buffer_states = self.system.transition_manufacturing()
        print("machine_states after transitioning: ", self.machine_states, "buffer_states after transitioning: ", self.buffer_states)
        # Define a dictionary to map machine states to observation values
        state_mapping = {'Off': 0, 'Opr': 1, 'Brk': 2, 'Sta': 3, 'Blo': 4}

        # Initialize the observation list
        observation = []

        # Iterate through machine states and add the corresponding observation value
        for idx, state in enumerate(self.machine_states):
            if state in state_mapping:
                observation.append(state_mapping[state])
            else:
                observation.append(0)  # Default to 'Off' state if unknown state is encountered

        # Append the buffer states to the observation
            observation.extend(self.buffer_states)
        # print("observation_disc",observation)
        # Actionsimulation.microgridActions_SolarWindGenerator
        action_sim= ActionSimulation(self.system)
        solar_power, wind_power, generator_power = action_sim.MicroGridActions_SolarWindGenerator(theta)
        self.actions_purchased, self.actions_discharged = action_sim.MicroGridActions_PurchasedDischarged(solar_actions, wind_actions, generator_actions)
        print("self.actions_discharged: ", self.actions_discharged)
        print("self.actions_purchased: ", self.actions_purchased)
        # print("solar power before conversion: ", solar_power, "type: ", type(solar_power))
        # Convert solar_power to a simple list of float values
        # solar_power_flat = [float(array[0]) for array in solar_power]
        # wind_power_flat = [float(array[0]) for array in wind_power]
        # generator_power_flat = [float(array[0]) for array in generator_power]
        # print("solar_power and its type: ", solar_power_flat, type(solar_power_flat))
        self.working_status, self.SOC = self.system.grid.transition(t)
        print("solar_power:", solar_power,"wind_power:", wind_power, "generator_power: ", generator_power, "SOC: ", self.SOC )

        # Construct the observation with SOC followed by the power source values
        observation.extend([self.SOC] + solar_power)
        observation.extend([self.SOC] + wind_power)
        observation.extend([self.SOC] + generator_power)

        # print("observation", observation)
        # Define the number of elements for the first five lists and subsequent lists
        number_machines = 5
        power_sources = 4 # power sources are actually 3, 4 is just for coding purpose below

        # Initialize the list of lists
        obs_list = []

        # Create the first five lists
        for i in range(0, number_machines * number_machines, number_machines):
            obs_list.append(observation[i:i + number_machines])

        # Create subsequent lists with 4 values each
        for i in range(number_machines * number_machines, len(observation), power_sources):
            obs_list.append(observation[i:i + number_machines - 1])
        # return machine_states, buffer_states, SOC, solar_power, wind_power, generator_power
        # print("obs_list: ", obs_list)
        return obs_list
    def get_reward(self, t):
        # print("rate_consumption_charge: ", rate_consumption_charge[self.t//8640], "self.t:", t)
        avg_cost= self.system.average_total_cost(rate_consumption_charge[t], t) #[t//8640]
        print("Reward avg cost: ", avg_cost)
        return avg_cost
    def is_done(self, t):
        # if t>=T:
        #     return [True, True, True, True, True, True, True, True]
        # else:
            return [False, False, False, False, False, False, False, False]
    def get_info(self):
        return "system_info"
