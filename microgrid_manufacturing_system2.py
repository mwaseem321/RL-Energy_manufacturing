# -*- coding: utf-8 -*-
"""
Created on Fri Jan 3 14:33:36 2020
@author: Wenqing Hu (Missouri S&T)
Title: MDP for joint control of microgrid and manufactoring system
"""
import math

import numpy as np
from random import choice
from projectionSimplex import projection

"""
Set up all parameters that are constant throughout the system
units of measurement: hour, km, MegaWatt(10^6Watt), 10^4 us dollar ($)
"""
Delta_t=1
#the actual time measured in one decision epoch unit, in hours#
cutin_windspeed=3*3.6
#the cut-in windspeed (km/h=1/3.6 m/s), v^ci#
cutoff_windspeed=11*3.6
#the cut-off windspeed (km/h=1/3.6 m/s), v^co#
rated_windspeed=7*3.6
#the rated windspeed (km/h=1/3.6 m/s), v^r#
charging_discharging_efficiency=0.95
#the charging-discharging efficiency, eta#
rate_battery_discharge=2/1000
#the rate for discharging the battery (MegaWatt), b#
unit_operational_cost_solar=0.17/10
#the unit operational and maintanance cost for generating power from solar PV (10^4$/MegaWattHour=10 $/kWHour), r_omc^s#
unit_operational_cost_wind=0.08/10
#the unit operational and maintanance cost for generating power from wind turbine (10^4$/MegaWattHour=10 $/kWHour), r_omc^w#
unit_operational_cost_generator=0.45/10
#the unit opeartional and maintanance cost for generating power from generator (10^4$/MegaWattHour=10 $/kWHour), r_omc^g#
unit_operational_cost_battery=0.9/10
#the unit operational and maintanance cost for battery storage system per unit charging/discharging cycle (10^4$/MegaWattHour=10 $/kWHour), r_omc^b#
capacity_battery_storage=350/1000
#the capacity of battery storage system (MegaWatt Hour=1000 kWHour), e#

# actual_capacity_Eat= 1 #Actual capacity of the battery

SOC_max=0.95*capacity_battery_storage
#the maximum state of charge of battery system#
SOC_min=0.05*capacity_battery_storage
#the minimum state of charge of battery system#
area_solarPV=1400/(1000*1000)
#the area of the solar PV system (km^2=1000*1000 m^2), a#
efficiency_solarPV=0.2
#the efficiency of the solar PV system, delta#
density_of_air=1.225
#calculate the rated power of the wind turbine, density of air (10^6kg/km^3=1 kg/m^3), rho#
radius_wind_turbine_blade=25/1000
#calculate the rated power of the wind turbine, radius of the wind turbine blade (km=1000 m), r#
average_wind_speed=3.952*3.6
#calculate the rated power of the wind turbine, average wind speed (km/h=1/3.6 m/s), v_avg (from the windspeed table)#
power_coefficient=0.593
#calculate the rated power of the wind turbine, power coefficient, theta#
gearbox_transmission_efficiency=0.9
#calculate the rated power of the wind turbine, gearbox transmission efficiency, eta_t#
electrical_generator_efficiency=0.9
#calculate the rated power of the wind turbine, electrical generator efficiency, eta_g#
rated_power_wind_turbine_original=0.5*density_of_air*np.pi*radius_wind_turbine_blade*radius_wind_turbine_blade*average_wind_speed*average_wind_speed*average_wind_speed*power_coefficient*gearbox_transmission_efficiency*electrical_generator_efficiency
rated_power_wind_turbine=rated_power_wind_turbine_original/(3.6*3.6*3.6)
#the rated power of the wind turbine, RP_w (MegaWatt=10^6 W), 
#with the radius_wind_turbine_blade measured in km=10^3m, average wind speed measured in km/hour=3.6m/s, RP_w will be calculated as RP_w_numerical
#then RP_w in MegaWatt=(1 kg/m^3)*(10^3 m)*(10^3 m)*(3.6 m/s)*(3.6 m/s)*(3.6 m/s)*RP_w_numerical=3.6^3*10^6 RP_w_numerical W=3.6^3 RP_w_numerical MegaWatt#
number_windturbine=1
#the number of wind turbine in the onsite generation system, N_w#
number_generators=1
#the number of generators, n_g#
rated_output_power_generator=65/1000
#the rated output power of the generator (MegaWatt=1000kW), G_p#
unit_reward_production=10000/10000
#the unit reward for each unit of production (10^4$/unit produced), i.e. the r^p, this applies to the end of the machine sequence#
unit_reward_soldbackenergy=0.2/10
#the unit reward from sold back energy (10^4$/MegaWattHour=10 $/kWHour), r^sb#
number_machines=5
#the total number of machines in the manufacturing system, total number of buffers=number_machines-1#
machine_lifetime_scale_parameter=[111.39/60, 51.1/60, 110.9/60, 239.1/60, 112.1/60]
#the set of machine lifetime scale parameters (hour), size=number_machines#
machine_lifetime_shape_parameter=[1.5766, 1.6532, 1.7174, 1.421, 1.591]
#the set of machine lifetime shape parameters, size=number_machines#
machine_repairtime_mean=[4.95/60, 11.7/60, 15.97/60, 27.28/60, 18.37/60]
#the set of machine repairtime mean parameters (hour), size=number_machines#
machine_power_consumption_Opr=[115.5/1000, 115.5/1000, 115.5/1000, 170.5/1000, 132/1000]
#the set of amount of power drawn (MegaWatt) by the machine if the machine state is Opr (Operating), size=number_machines#
machine_power_consumption_Idl=[105/1000, 105/1000, 105/1000, 155/1000, 120/1000]
#the set of amount of power drawn (MegaWatt) by the machine if the machine state is Sta (Starvation) or Blo (Blockage), both are Idl (Idle) states, size=number_machines#
list_buffer_max=[1000, 1000, 1000, 1000]
list_buffer_min=[0, 0, 0, 0]
#the maximum and minumum of buffers, size=number_machine-1#



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

"""
Define 3 major classes in the system: Machine, Buffer, Microgrid
"""
"""
the Machine class defines the variables and functions of one machine
"""

class Machine(object):
    def __init__(self,
                 name=1,
                 #the label of this machine#
                 lifetime_shape_parameter=0, 
                 #random lifetime of machine follows Weibull distribution with shape parameter lifetime_shape_parameter
                 lifetime_scale_parameter=0,
                 #random lifetime of machine follows Weibull distribution with scale parameter lifetime_scale_parameter
                 repairtime_mean=0,
                 #random repair time of machine follows exponential distribution with mean repairtime_mean
                 power_consumption_Opr=0,
                 #amount of power drawn by the machine if the machine state is Opr (Operating)
                 power_consumption_Idl=0,
                 #amount of power drawn by the machine if the machine state is Sta (Starvation) or Blo (Blockage), both are Idl (Idle) states
                 state="OFF",
                 #machine state can be "Opr" (Operating), "Blo" (Blockage), "Sta" (Starvation), "Off", "Brk" (Break)
                 control_action="K",
                 #control actions of machine, actions can be "K"-action (keep the original operational), "H"-action (to turn off the machine) or "W"-action (to turn on the machine)#
                 is_last_machine=False
                 #check whether or not the machine is the last machine in the queue, if it is last machine, then it contributes to the throughput#
                 ):
        self.name=name
        self.lifetime_shape_parameter=lifetime_shape_parameter
        self.lifetime_scale_parameter=lifetime_scale_parameter
        self.repairtime_mean=repairtime_mean
        self.power_consumption_Opr=power_consumption_Opr
        self.power_consumption_Idl=power_consumption_Idl
        self.unit_reward_production=unit_reward_production
        self.state=state
        self.control_action=control_action
        self.is_last_machine=is_last_machine
    
    def EnergyConsumption(self):
        #Calculate the energy consumption of one machine in a time unit#
        PC=0 
        #PC is the amount drawn by a machine in a time unit#
        if self.state== 2 or self.state==0: #(2 or "Brk"),  (0 or "Off")
            PC=0
        elif self.state== 1: #(1 or "Opr")
            PC=self.power_consumption_Opr*Delta_t
        elif self.state== 3 or self.state== 4: # (3 or "Sta"), (4 or "Blo")
            PC=self.power_consumption_Idl*Delta_t
        # print(f"machine's state is {self.state} in the machine energy consumption function")
        # print("Energy consumption PC of each machine,:", PC)
        return PC

    def LastMachineProduction(self):
        #only the last machine will produce that contributes to the throughput, when the state is Opr and the control action is K#
        # print("self.last_machine_status: ", self.is_last_machine)
        # print("self.state in last machine:", self.state, "self.control_action: ", self.control_action)
        if self.is_last_machine:
            if self.state!=  1 or self.control_action=="H": # ("Opr" or 1)
                throughput=0
            elif self.state== 1 and self.control_action=="K": #("Opr" or 1)
                throughput=1
            else:
                throughput=0
        else:
            throughput=0
        print("throughout returned in lastMachine: ", throughput)
        return throughput
    
    def NextState_IsOff(self):
        #Based on the current state of the machine, determine if the state of the machine at next decision epoch is "Off"#
        #If is "Off" return True otherwise return False#
        #When return False, the next state lies in the set {"Brk", "Opr", "Sta", "Blo"}#
        if self.state== 0:
            if self.control_action!="W":
                IsOff=True
            else:
                IsOff=False
        else:
            if self.control_action=="H":
                IsOff=True
            else:
                IsOff=False
        return IsOff
            
    def NextState_IsBrk(self):
        #Based on the current state of the machine, determine if the state of the machine at next decision epoch is "Brk"#
        #If is "Brk" return True otherwise return False#
        #When return False, the next state lies in the set {"Opr", "Sta", "Blo", "Off"}#
        L=self.lifetime_scale_parameter*np.random.weibull(self.lifetime_shape_parameter, 1)
        #the random variable L is the lifetime#
        D=np.random.exponential(self.repairtime_mean)
        #the random variable D is the repair time# 
        if self.state== 2:
            if D>=Delta_t:
                IsBrk=True
            else:
                IsBrk=False
        else:
            if self.state!= 0:
                if L<Delta_t:
                    IsBrk=True
                else:
                    IsBrk=False
            else:
                IsBrk=False
        return IsBrk
    
    def PrintMachine(self, file):
        #print the status of the current machine: state, control_action taken, Energy Consumption, throughput, decide whether the next machine state is Brk#
        print("Machine", self.name, "=", self.state, ",", "action=", self.control_action, file=file)
        print(" Energy Consumption=", self.EnergyConsumption(), file=file)
        if self.is_last_machine:
            print(" ", file=file)
            print(" throughput=", self.LastMachineProduction(), file=file)
            print("\n", file=file)
        return None
        
        
        
"""
the Buffer class defines variables and functions of one buffer
"""
class Buffer(object):
    def __init__(self, 
                 name=1,
                 #the label of this buffer#
                 state=0,
                 #the buffer state is an integer from buffer_min (=0) to buffer_max 
                 buffer_max=1000,
                 #the maximal capacity of the buffer#
                 buffer_min=0,
                 #the minimal capacity of the buffer is zero#
                 previous_machine_state="Opr",
                 #the state of the machine that is previous to the current buffer#
                 next_machine_state="Off",
                 #the state of the machine that is next to the current buffer#
                 previous_machine_control_action="K",
                 #the control action applied to the machine that is previous to the current buffer#
                 next_machine_control_action="K"
                 #the control action applied to the machine that is next to the current buffer#
                 ):
        self.name=name
        self.state=state
        self.buffer_max=buffer_max
        self.buffer_min=buffer_min
        self.previous_machine_state=previous_machine_state
        self.next_machine_state=next_machine_state
        self.previous_machine_control_action=previous_machine_control_action
        self.next_machine_control_action=next_machine_control_action
        
    def NextState(self):
        #calculate the state of the buffer at next decision epoch, return this state#
        nextstate=self.state
        if self.previous_machine_state!=1 or self.previous_machine_control_action== ("H"): #(1 or "Opr")
            I_previous=0
        elif self.previous_machine_state==1 and self.previous_machine_control_action== ("K"): #(1 or "Opr")
            I_previous=1
        else:
            I_previous=0
        if self.next_machine_state!= 1 or self.next_machine_control_action=="H": #(1 or "Opr")
            I_next=0
        elif self.next_machine_state== 1 and self.next_machine_control_action=="K": #(1 or "Opr")
            I_next=1
        else:
            I_next=0
        nextstate=nextstate+I_previous-I_next
        if nextstate>self.buffer_max:
            nextstate=self.buffer_max
        if nextstate<self.buffer_min:
            nextstate=self.buffer_min
        return nextstate

    def PrintBuffer(self, file):
        #print the status of the current buffer: buffer state, next buffer state#
        print("Buffer", self.name, "=", self.state, file=file)
        print("\n", file=file)
        return None


        
"""
the Microgrid class defines variables and functions of the microgrid
"""
class Microgrid(object):
    def __init__(self,
                 workingstatus, #=[0,0,0],
                 #the working status of [solar PV, wind turbine, generator]#
                 SOC, #=0,
                 #the state of charge of the battery system#
                 actions_adjustingstatus, #=[0,0,0],
                 #the actions of adjusting the working status (connected =1 or not =0 to the load) of the [solar, wind, generator]#
                 actions_solar, #=[0,0,0],
                 #the solar energy used for supporting [manufaturing, charging battery, sold back]#
                 actions_wind, #=[0,0,0],
                 #the wind energy used for supporting [manufacturing, charging battery, sold back]#
                 actions_generator, #=[0,0,0],
                 #the use of the energy generated by the generator for supporting [manufacturing, charging battery, sold back]#
                 actions_purchased, #=[0,0],
                 #the use of the energy purchased from the grid for supporting [manufacturing, charging battery]#
                 actions_discharged, #=0,
                 #the energy discharged by the battery for supporting manufacturing#
                 solarirradiance, #=0,
                 #the environment feature: solar irradiance at current decision epoch#
                 windspeed, #=0,
                 #the environment feature: wind speed at current decision epoch#
                 actual_capacity=capacity_battery_storage, # Muhammad
                 battery_rated_capacity=1, # 90kWh, 0.09Mwh Muhammad
                 t=0, # Muhammad
                 f=0,#Mihitha
                 C_Dt=[0]*10000,#Mihitha
                 EbT=[],#Muhammad
                 EAT=[0]*10000 ,#Mihitha
                 AC=[0.35] + [0]*10000 ,#Mihitha # updated battery capacity at each time
                 Dt=0,
                 
                 ): 
        self.workingstatus=workingstatus
        self.SOC=SOC
        self.actions_adjustingstatus=actions_adjustingstatus
        self.actions_solar=actions_solar
        self.actions_wind=actions_wind
        self.actions_generator=actions_generator
        self.actions_purchased=actions_purchased
        self.actions_discharged=actions_discharged
        self.solarirradiance=solarirradiance
        self.windspeed=windspeed
        self.actual_capacity= actual_capacity #Muhammad
        self.battery_rated_capacity= battery_rated_capacity # Muhammad
        self.Ebr = self.actual_capacity / self.battery_rated_capacity
        self.t= t
        self.f=f#Mihitha
        self.EbT= EbT
        self.EAT= EAT #Mihitha
        self.C_Dt=C_Dt
        self.SOC_min=0
        self.Dt=Dt #Mihitha
        self.AC=AC #Mihitha
        self.EAT_plot= []
        self.Dt_plot=[]
        self.time_steps_plot= []
    def get_actual_capacity_Eat(self, t):  # Muhammad
        # print("Before calling battery_life cycle Ct at t: ", t)
        alpha_0 = 4980 # Need to update these values
        alpha_1 = 1.98
        alpha_2 = 0.016
        # print("Before calling depth of discharge at t: ", t)
        # print("self.actions_solar[1]: ", self.actions_solar[1])
        # print("self.actions_generator[1]: ", self.actions_generator[1])
        # print("self.actions_wind[1]: ", self.actions_wind[1])
        # print("self.actions_purchased[1]: ", self.actions_purchased[1])
        # print("self.actions_discharged[1]: ", self.actions_discharged)

        #Charging/discharging power EbT
        if self.actions_solar[1] > 0 or self.actions_generator[1] > 0 or self.actions_wind[1] > 0 or \
                self.actions_purchased[1] >0:
            self.EbT.append(1)
        elif self.actions_discharged > 0:
            self.EbT.append(-1)
        else:
            self.EbT.append(0)
        # print("self.EbT: ", self.EbT)
        

        # Updating switching cycles counter k according to that value for f
        if t != 0:
            # print("self.EbT", self.EbT)
            k_status = self.EbT[t] * self.EbT[t - 1]
            if k_status == -1:
                self.f = 1
            elif k_status == 1: #Mihitha
                self.f = 0   #Mihitha
            elif k_status == 0:
                self.f = 1 


        #Depth of discharge
        if t!=0: 
        #Accumulated energy after charging or discharging cycle continues
         self.EAT[t] = (1-self.f)*self.EAT[t-1] + (self.actions_solar[2 - 1] + self.actions_wind[2 - 1] + self.actions_generator[2 - 1] +
                        self.actions_purchased[
                            2 - 1]) * charging_discharging_efficiency - self.actions_discharged / charging_discharging_efficiency    
         
         # print("self.AC[t]: ", self.AC[t])
         # print("self.EAT[t]: ", self.EAT[t])
         self.EAT_plot.append(self.EAT[t])
         self.time_steps_plot.append(t)
         self.Dt=abs(self.EAT[t] / self.AC[t])
         # print("self.Dt: ", self.Dt)
         self.Dt_plot.append(self.Dt)
         #if t!=0:
          #self.Dt=abs(self.EAT[t]) / self.AC[t-1] 
         # print("self.EAT: ", self.EAT)



        # Battery life cycle at t C(Dt)

        if self.Dt != 0:
            self.C_Dt[t] = alpha_0 * (self.Dt ** -alpha_1) * math.exp(-alpha_2 * self.Dt)
        else:
            self.C_Dt[t] = 0
        # print("C_Dt: ", self.C_Dt )

        # Battery actual capacity Eat
        # self.Ebr= self.actual_capacity # why were we taking Ebr=actual capacity, while it is divide by rated capacity as well
        if self.C_Dt[t]!= 0 and self.f==0 and self.C_Dt[t-1]!= 0:
            #updated_capacity= self.actual_capacity - (self.Ebr/(2 * self.C_Dt))
            #self.actual_capacity= updated_capacity
            # print("self.AC[t] for next AC:", self.AC[t],"self.C_Dt[t] is: ", self.C_Dt[t], "self.C_Dt[t-1]: ", self.C_Dt[t-1], "time is: ", t)
            self.AC[t+1]= self.AC[t] - (self.Ebr/(2 * self.C_Dt[t])) + (self.Ebr/(2 * self.C_Dt[t-1]))
        elif self.f==1 and self.C_Dt[t] != 0 :
            #self.actual_capacity= self.actual_capacity
            self.AC[t+1]=self.AC[t] - (self.Ebr/(2 * self.C_Dt[t])) 
        else:
            self.AC[t+1]= self.AC[t]
        # print("self.actual_capacity AC: ", self.AC)
        return self.actual_capacity

    def get_SOC_min(self, t):  # Muhammad
        # print("SoC_min is called here at t: ", t)
        # print("self.AC[t] in SOC_min is: ", self.AC[t])
        SOC_min= 0.05*self.AC[t]
        return SOC_min

    def get_SOC_max(self, t):  # Muhammad
        # print("SoC_max is called at time: ", t)
        # print(f"self.AC[t] in SOC_max is {self.AC[t]} at time step {t}")
        SOC_max = 0.95 * self.AC[t]
        # print("SOC Max: ", SOC_max)
        return SOC_max

    def get_battery_degradation_cost(self, t):  # Muhammad
        C_cap= 33.4 #2400 # capital cost of the battery i.e., initial installation cost
        # C_Dt= self.get_battery_life_cycle_Ct(t)
        if self.f==0 and self.C_Dt[t]!=0 and self.C_Dt[t-1]!=0:
            # print("self.C_Dt[t]: ", self.C_Dt[t])
            # print("self.C_Dt[t-1]: ", self.C_Dt[t-1])
            # print("self.Ebr: ", self.Ebr)
            C_B= (C_cap*self.Ebr)/(2*self.C_Dt[t])-(C_cap*self.Ebr)/(2*self.C_Dt[t-1])
        elif self.f==1 and self.C_Dt[t]!=0:
            C_B = (C_cap * self.Ebr) / (2 * self.C_Dt[t])
        # return C_B
            
        else:
            C_B= 0

        return C_B


    def transition(self, t):
        # print("transition function is called at t: ", t)
        workingstatus=self.workingstatus
        # SOC=self.SOC
        print("self.actions_adjustingstatus in the transition of grid: ", self.actions_adjustingstatus)
        if self.actions_adjustingstatus[1-1]==1:
            workingstatus[1-1]=1
        else:
            workingstatus[1-1]=0
        #determining the next decision epoch working status of solar PV, 1=working, 0=not working#
        if self.actions_adjustingstatus[2-1]==0 or self.windspeed>cutoff_windspeed or self.windspeed<cutin_windspeed:
            workingstatus[2-1]=0
        else: 
            if self.actions_adjustingstatus[2-1]==1 and self.windspeed<=cutoff_windspeed and self.windspeed>=cutin_windspeed:
                workingstatus[2-1]=1
        #determining the next decision epoch working status of wind turbine, 1=working, 0=not working#        
        if self.actions_adjustingstatus[3-1]==1:
            workingstatus[3-1]=1
        else:
            workingstatus[3-1]=0
        #determining the next decision epoch working status of generator, 1=working, 0=not working#
        # SOC=self.SOC+(self.actions_solar[2-1]+self.actions_wind[2-1]+self.actions_generator[2-1]+self.actions_purchased[2-1])*charging_discharging_efficiency-self.actions_discharged/charging_discharging_efficiency
        self.SOC = self.SOC + (self.actions_solar[2 - 1] + self.actions_wind[2 - 1] + self.actions_generator[2 - 1] +
                          self.actions_purchased[
                              2 - 1]) * charging_discharging_efficiency - self.actions_discharged / charging_discharging_efficiency
        # print("EbT Before SoC_max: ", self.EbT)
        # print("SOC calculated in the transition of microgrid class: ", self.SOC)
        self.actual_capacity= self.get_actual_capacity_Eat(t)
        self.SOC_max= self.get_SOC_max(t) #Muhammad
        # print("self.SOC_max: ", self.SOC_max)
        # print("self.SOC_min: ", self.SOC_min)
        self.SOC_min= self.get_SOC_min(t) # Muhammad
        if self.SOC>self.SOC_max:
            self.SOC=self.SOC_max
        if self.SOC<self.SOC_min:
            self.SOC=self.SOC_min
        #determining the next desicion epoch SOC, state of charge of the battery system#
        return workingstatus, self.SOC
    
    def EnergyConsumption(self):
        #returns the energy consumption from the grid#
        return -(self.actions_solar[1-1]+self.actions_wind[1-1]+self.actions_generator[1-1]+self.actions_discharged)

    def energy_generated_solar(self):
        #calculate the energy generated by the solar PV, e_t^s#
        print("self.workingstatus in the energy_generated_solar: ", self.workingstatus)
        if self.workingstatus[1-1]==1:
            energy_generated_solar=self.solarirradiance*area_solarPV*efficiency_solarPV/1000
        else:
            energy_generated_solar=0
        return energy_generated_solar
    
    def energy_generated_wind(self):
        #calculate the energy generated by the wind turbine, e_t^w#
        if self.workingstatus[2-1]==1 and self.windspeed<rated_windspeed and self.windspeed>=cutin_windspeed:
            energy_generated_wind=number_windturbine*rated_power_wind_turbine*(self.windspeed-cutin_windspeed)/(rated_windspeed-cutin_windspeed)
        else:
            if self.workingstatus[2-1]==1 and self.windspeed<cutoff_windspeed and self.windspeed>=rated_windspeed:
                energy_generated_wind=number_windturbine*rated_power_wind_turbine*Delta_t
            else:
                energy_generated_wind=0
        return energy_generated_wind
    
    def energy_generated_generator(self):
        #calculate the energy generated bv the generator, e_t^g#
        if self.workingstatus[3-1]==1:
            energy_generated_generator=number_generators*rated_output_power_generator*Delta_t
        else:
            energy_generated_generator=0
        return energy_generated_generator
        
    def OperationalCost(self, t):
        #returns the operational cost for the onsite generation system#
        if self.workingstatus[1-1]==1:
            energy_generated_solar=self.solarirradiance*area_solarPV*efficiency_solarPV/1000
        else:
            energy_generated_solar=0
        #calculate the energy generated by the solar PV, e_t^s#
        if self.workingstatus[2-1]==1 and self.windspeed<rated_windspeed and self.windspeed>=cutin_windspeed:
            energy_generated_wind=number_windturbine*rated_power_wind_turbine*(self.windspeed-cutin_windspeed)/(rated_windspeed-cutin_windspeed)
        else:
            if self.workingstatus[2-1]==1 and self.windspeed<cutoff_windspeed and self.windspeed>=rated_windspeed:
                energy_generated_wind=number_windturbine*rated_power_wind_turbine*Delta_t
            else:
                energy_generated_wind=0
        #calculate the energy generated by the wind turbine, e_t^w#
        if self.workingstatus[3-1]==1:
            energy_generated_generator=number_generators*rated_output_power_generator*Delta_t
        else:
            energy_generated_generator=0

        # calculate the energy generated bv the generator, e_t^g#
        Battery_degradation_cost= self.get_battery_degradation_cost(t)  # Muhammad
        print("Battery degradation cost in main ftn is: ", Battery_degradation_cost)
        operational_cost=energy_generated_solar*unit_operational_cost_solar+energy_generated_wind*unit_operational_cost_wind+energy_generated_generator*unit_operational_cost_generator
        # print("operational_cost without battery degradation: ", operational_cost)
        operational_cost += Battery_degradation_cost  # Muhammad
        # print("operational_cost with battery degradation: ", operational_cost)
        # operational_cost+=(self.actions_discharged+self.actions_solar[2-1]+self.actions_wind[2-1]+self.actions_generator[2-1])*Delta_t*unit_operational_cost_battery/(2*capacity_battery_storage*(SOC_max-SOC_min))
        #calculate the operational cost for the onsite generation system#
        return operational_cost
    
    def SoldBackReward(self):
        #calculate the sold back reward (benefit)#
        return (self.actions_solar[3-1]+self.actions_wind[3-1]+self.actions_generator[3-1])*unit_reward_soldbackenergy
    
    def PrintMicrogrid(self, file, t):
        #print the current and the next states of the microgrid#
        print("Microgrid working status [solar PV, wind turbine, generator]=", self.workingstatus, ", SOC=", self.SOC, file=file)
        print(" microgrid actions [solar PV, wind turbine, generator]=", self.actions_adjustingstatus, file=file)
        print(" solar energy supporting [manufaturing, charging battery, sold back]=", self.actions_solar, file=file)
        print(" wind energy supporting [manufacturing, charging battery, sold back]=", self.actions_wind, file=file)
        print(" generator energy supporting [manufacturing, charging battery, sold back]=", self.actions_generator, file=file)
        print(" energy purchased from grid supporting [manufacturing, charging battery]=", self.actions_purchased, file=file)
        print(" energy discharged by the battery supporting manufacturing=", self.actions_discharged, file=file)
        print(" solar irradiance=", self.solarirradiance, file=file)
        print(" wind speed=", self.windspeed, file=file)
        print(" Microgrid Energy Consumption=", self.EnergyConsumption(), file=file)
        print(" Microgrid Operational Cost=", self.OperationalCost(t), file=file)
        print(" Microgrid SoldBackReward=", self.SoldBackReward(), file=file)
        print("\n", file=file)
        return None


"""    
Combining the above three classes, define the variables and functions for the whole manufacturing system
"""
class ManufacturingSystem(object):
    def __init__(self,
                 machine_states,
                 #set the machine states for all machines in the manufacturing system#
                 machine_control_actions,
                 #set the control actions for all machines in the manufacturing system#
                 buffer_states,
                 #set the buffer states for all buffers in the manufacturing system#
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
                 #set the microgrid states and control actions#
                 ):
        self.machine_states=machine_states
        self.machine_control_actions=machine_control_actions
        self.buffer_states=buffer_states
        #initialize all machines, ManufacturingSystem.machine=[Machine1, Machine2, ..., Machine_{number_machines}]#
        self.machine=[]
        for i in range(number_machines):
            if i!=number_machines-1:
                self.machine.append(Machine(name=i+1, 
                                            state=self.machine_states[i], 
                                            lifetime_shape_parameter=machine_lifetime_shape_parameter[i],
                                            lifetime_scale_parameter=machine_lifetime_scale_parameter[i],
                                            repairtime_mean=machine_repairtime_mean[i],
                                            power_consumption_Opr=machine_power_consumption_Opr[i],
                                            power_consumption_Idl=machine_power_consumption_Idl[i],                                            
                                            control_action=self.machine_control_actions[i], 
                                            is_last_machine=False))
            else:
                self.machine.append(Machine(name=i+1, 
                                            state=self.machine_states[i], 
                                            lifetime_shape_parameter=machine_lifetime_shape_parameter[i],
                                            lifetime_scale_parameter=machine_lifetime_scale_parameter[i],
                                            repairtime_mean=machine_repairtime_mean[i],
                                            power_consumption_Opr=machine_power_consumption_Opr[i],
                                            power_consumption_Idl=machine_power_consumption_Idl[i],                                            
                                            control_action=self.machine_control_actions[i], 
                                            is_last_machine=True))
        #initialize all buffers, ManufacturingSystem.buffer=[Buffer1, Buffer2, ..., Buffer_{numbers_machines-1}]
        self.buffer=[]
        for j in range(number_machines-1):
            self.buffer.append(Buffer(name=j+1, 
                                      state=self.buffer_states[j], 
                                      buffer_max=list_buffer_max[j],
                                      buffer_min=list_buffer_min[j],
                                      previous_machine_state=self.machine[j].state, 
                                      next_machine_state=self.machine[j+1].state,
                                      previous_machine_control_action=self.machine[j].control_action,
                                      next_machine_control_action=self.machine[j+1].control_action
                                      ))
        self.grid=grid
        
    def transition_manufacturing(self):
        #based on current states and current control actions of the whole manufacturing system, calculate states at the the next decision epoch#
        #states include machine states, buffer states and microgrid states#
        buffer_states=[]
        for j in range(number_machines-1):
            buffer_states.append(self.buffer[j].NextState())
        # print("buffer states in transition_manuf: ", buffer_states)
        #based on current machine states and control actions taken, calculate the next states of all buffers#
        Off=[]
        Brk=[]
        Sta=[]
        Blo=[]
        #Set up four 0/1 sequence that test the next states being "Off", "Brk", "Sta" or "Blo". If none of these, then "Opr"#
        for i in range(number_machines):
            Off.append(0)
            Brk.append(0)
            Sta.append(0)
            Blo.append(0)
        for i in range(number_machines):
        #Check the possibilities of "Off" or "Brk" states#    
            if self.machine[i].NextState_IsOff():
                Off[i]=1
            if self.machine[i].NextState_IsBrk():
                Brk[i]=1
        for i in range(number_machines):
        #Check the possibilities of "Sta" states#
            if i==0:
                Sta[i]=0
            else:
                if Brk[i]==1 or Off[i]==1:
                    Sta[i]=0
                else:
                    if buffer_states[i-1]==self.buffer[i-1].buffer_min:
                        if Brk[i-1]==1 or Sta[i-1]==1 or Off[i-1]==1:
                            Sta[i]=1
                        else:
                            Sta[i]=0
                    else:
                        Sta[i]=0
        for i in reversed(range(number_machines)):
        #Check the possibilities of "Blo" states#
            if i==number_machines-1:
                Blo[i]=0
            else:
                if Brk[i]==1 or Off[i]==1:
                    Blo[i]=0
                else:
                    if buffer_states[i]==self.buffer[i].buffer_max:
                        if Brk[i+1]==1 or Blo[i+1]==1 or Off[i+1]==1:
                            Blo[i]=1
                        else:
                            Blo[i]=0
                    else:
                        Blo[i]=0
        #based on current machine states and control actions taken, calculate the next states of all machines#    
        machine_states=[]                
        for i in range(number_machines):
            if Off[i]==1:
                machine_states.append("Off")
            elif Brk[i]==1:
                machine_states.append("Brk")
            elif Sta[i]==1:
                machine_states.append("Sta")
            elif Blo[i]==1:
                machine_states.append("Blo")
            else: 
                machine_states.append("Opr")
        #return the new states#
        return machine_states, buffer_states

    def average_total_cost(self, current_rate_consumption_charge, t):
        #calculate the average total cost of the manufacturing system, E(S,A), based on the current machine, buffer, microgrid states and actions#
        # print("current_rate_consumption_charge: ", current_rate_consumption_charge)
        E_mfg=0
        #total energy consumed by the manufacturing system, summing over all machines#
        for i in range(number_machines):
            E_mfg=E_mfg+self.machine[i].EnergyConsumption()
        #the energy consumption cost#            
        TF=(E_mfg+self.grid.EnergyConsumption())*current_rate_consumption_charge
        #the operational cost for the microgrid system#
        MC=self.grid.OperationalCost(t)
        #the prduction throughput of the manufacturing system#
        # TP=self.machine[number_machines-1].LastMachineProduction()*unit_reward_production
        # print("Throughput: ", TP)
        #the sold back reward#
        TP= self.throughput()
        SB=self.grid.SoldBackReward()
        print("Throughput: ", TP)
        print("SOld Back:  ", SB)
        print("Energy consumption cost:  ", TF)
        print("operational cost, MC:  ", MC)
        return TP+SB-TF-MC #TF+MC-TP-SB
    
    def energydemand(self, current_rate_consumption_charge):
        #calculate the total energy demand TF of the system, based on the current machine, buffer, microgrid states and actions#
        E_mfg=0
        #total energy consumed by the manufacturing system, summing over all machines#
        for i in range(number_machines):
            E_mfg=E_mfg+self.machine[i].EnergyConsumption()
        #the energy consumption cost#            
        TF=(E_mfg+self.grid.EnergyConsumption())*current_rate_consumption_charge
        return TF
    
    def throughput(self):
        #calculate total throughput TP of the manufacturing system, based on the current machine, buffer, microgrid states and actions#
        #the prduction throughput of the manufacturing system#
        TP=self.machine[number_machines-1].LastMachineProduction()*unit_reward_production
        return TP 

    def PrintSystem(self, file, timepoint):
        for i in range(number_machines):
            self.machine[i].PrintMachine(file)
            if i!=number_machines-1:
                self.buffer[i].PrintBuffer(file)
        self.grid.PrintMicrogrid(file, timepoint)
        print("Average Total Cost=", self.average_total_cost(rate_consumption_charge[timepoint//8640], timepoint), file=file)
        print("\n", file=file)
        return None
       



"""
Simulate admissible actions based on the current state S_{t+1} of the manufacturing system, 
the admissible actions are A_{t+1}=(A^d, A^c, A^r)
"""
class ActionSimulation(object):
    def __init__(self,
                 System=ManufacturingSystem(machine_states=["Off" for _ in range(number_machines)],
                                            machine_control_actions=["K" for _ in range(number_machines)],
                                            buffer_states=[0 for _ in range(number_machines-1)],
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
                                                           ))
                 ):
        #the ManufacturingSystem is with new states S_{t+1} but old actions A_{t}, we obtain the admissible A_{t+1} in this class#
        self.System=System
        # print("self.system.machine_states in Action_simulation class: ", self.System.machine_states)
    def MachineActions(self):
        #Based on current machine states in the system, randomly uniformly simulate an admissible action for all machines#
        machine_actions=[]
        for i in range(number_machines):
            if self.System.machine_states[i]==1: #("Opr" or 1)
                machine_actions.append(choice(["K", "H"]))
            elif self.System.machine_states[i]==4: #("Blo" or 4)
                machine_actions.append(choice(["K", "H"]))
            elif self.System.machine_states[i]==3: # ("Sta"or 3)
                machine_actions.append(choice(["K", "H"]))
            elif self.System.machine_states[i]==0: #("Off" or 0)
                machine_actions.append(choice(["K", "W"]))
            else:
                machine_actions.append("K")
        return machine_actions
    
    def MicroGridActions_adjustingstatus(self):
        #randomly uniformly simulate an action that adjusts the status (connected=1) of the microgrid [solar, wind, generator]#
        actions_adjustingstatus=[]
        for i in range(3):
            actions_adjustingstatus.append(choice([0,1]))
        return actions_adjustingstatus
    
    def MicroGridActions_SolarWindGenerator(self, theta):
        #from the updated proportionality parameter theta return the corresponding actions on solar, wind and generator#
        #theta is the proportionality parameters theta=[lambda_s^m, lambda_s^b, lambda_w^m, lambda_w^b, lambda_g^m, lambda_g^]#
        #calculate the energy generated by the solar PV, e_t^s#
        # print("theta: ", theta)
        energy_generated_solar=self.System.grid.energy_generated_solar()
        #calculate the energy generated by the wind turbine, e_t^w#
        energy_generated_wind=self.System.grid.energy_generated_wind()
        #calculate the energy generated bv the generator, e_t^g#
        energy_generated_generator=self.System.grid.energy_generated_generator()
        #given the new theta, calculated the actions_solar, actions_wind, actions_generator#
        # actions_solar=[energy_generated_solar*theta[1-1], energy_generated_solar*theta[2-1], energy_generated_solar*(1-theta[1-1]-theta[2-1])]
        # actions_wind=[energy_generated_wind*theta[3-1], energy_generated_wind*theta[4-1], energy_generated_wind*(1-theta[3-1]-theta[4-1])]
        # actions_generator=[energy_generated_generator*theta[5-1], energy_generated_generator*theta[6-1], energy_generated_generator*(1-theta[5-1]-theta[6-1])]

        actions_solar = [energy_generated_solar * theta[0], energy_generated_solar * theta[1],
                         energy_generated_solar * (theta[2])]
        actions_wind = [energy_generated_wind * theta[3], energy_generated_wind * theta[4],
                        energy_generated_wind * (theta[5])]
        actions_generator = [energy_generated_generator * theta[6], energy_generated_generator * theta[7],
                             energy_generated_generator * (theta[8])]
        # print(f"actions generated in Action_simulation's MicrogridActionSolarWindGenerator function: actions_solar: {actions_solar}, actions_wind: {actions_wind}, actions_generator: {actions_generator}")
        return actions_solar, actions_wind, actions_generator
    
    def MicroGridActions_PurchasedDischarged(self, 
                                             actions_solar=[0,0,0],
                                             actions_wind=[0,0,0],
                                             actions_generator=[0,0,0]):
        #randomly simulate an action that determines the use of the purchased energy and the energy discharge#
        #actions_solar, actions_wind, actions_generator are the actions to be taken at current system states#
        # print(f"Actions provided as input in the MicroGridActions_PurchasedDischarged: {actions_solar}, {actions_wind}, {actions_generator}")
        TotalSoldBack=actions_solar[3-1]+actions_wind[3-1]+actions_generator[3-1]
        # print("TotalSoldBack: ", TotalSoldBack)
        #Total amount of sold back energy#
        TotalBattery=actions_solar[2-1]+actions_wind[2-1]+actions_generator[2-1]
        # print("TotalBattery: ", TotalBattery)
        #Total amount if energy charged to the battery#
        self.SOC_min= self.System.grid.SOC_min  # Muhammad
        # print("SOC_min before SOC_codition: ", self.SOC_min)
        SOC_Condition=self.System.grid.SOC-rate_battery_discharge*Delta_t/charging_discharging_efficiency-self.SOC_min
        #The condition for SOC at the current system state#
        # print("SOC_Condition: ", SOC_Condition)
        E_mfg=0
        for i in range(number_machines):
            E_mfg=E_mfg+self.System.machine[i].EnergyConsumption()
            # print(f"Machine {i} energy consumption is: {E_mfg}")
        #total energy consumed by the manufacturing system, summing over all machines#
        # print("E_mfg after")
        p_hat=E_mfg-(actions_solar[1-1]+actions_wind[1-1]+actions_generator[1-1])
        # print(f"total energy consumed by manuf_system p_hat: {p_hat}")
        if p_hat<0:
            p_hat=0
        #Set the p_hat#
        p_tilde=E_mfg-(actions_solar[1-1]+actions_wind[1-1]+actions_generator[1-1]+rate_battery_discharge*Delta_t)
        # print("p_tilde: ", p_tilde)
        if p_tilde<0:
            p_tilde=0
        #Set the p_tilde#
        ####Calculate actions_purchased and actions_discharged according to the table in the paper####
        actions_purchased=[0,0]
        actions_discharged=0
        if TotalSoldBack>0 and TotalBattery>0 and SOC_Condition>0:
            actions_purchased=[0,0]
            actions_discharged=0
        elif TotalSoldBack>0 and TotalBattery>0 and SOC_Condition<=0:
            actions_purchased=[0,0]
            actions_discharged=0
        elif TotalSoldBack>0 and TotalBattery<=0 and SOC_Condition>0:
            actions_purchased=[0,0]
            actions_discharged=choice([0, rate_battery_discharge*Delta_t])
        elif TotalSoldBack>0 and TotalBattery<=0 and SOC_Condition<=0:
            actions_purchased=[0,0]
            actions_discharged=0
        elif TotalSoldBack<=0 and TotalBattery>0 and SOC_Condition>0:
            actions_purchased[2-1]=choice([0, p_hat])
            actions_purchased[1-1]=p_hat-actions_purchased[2-1]
            actions_discharged=0
        elif TotalSoldBack<=0 and TotalBattery>0 and SOC_Condition<=0:
            actions_purchased[2-1]=choice([0, p_hat])
            actions_purchased[1-1]=p_hat-actions_purchased[2-1]
            actions_discharged=0
        elif TotalSoldBack<=0 and TotalBattery<=0 and SOC_Condition>0:
            actions_discharged=choice([0, rate_battery_discharge*Delta_t])
            if actions_discharged==0:
                actions_purchased[2-1]=choice([0, p_hat])
                actions_purchased[1-1]=p_hat-actions_purchased[2-1]
            else:
                actions_purchased[2-1]=0
                actions_purchased[1-1]=p_tilde
        else:
            actions_purchased[2-1]=choice([0, p_hat])
            actions_purchased[1-1]=p_hat-actions_purchased[2-1]
            actions_discharged=0
        #return actions_purchased and actions_discharged#

        return actions_purchased, actions_discharged
            


"""
Generate the set of all admissible microgrid actions for adjusting the microgrid status
Generate the set of all admissible microgrid actions for energy purchased/discharged , i.e. the remainder action A^r, 
 based on the current state S_{t+1} of the manufacturing system and the current discrete actions A^d 
Return all admissible microgrid actions for adjusting the microgrid status and all microgrid actions 
 for energy purchase/discharge as a list
"""
class MicrogridActionSet_Discrete_Remainder(object):
    def __init__(self,
                 System=ManufacturingSystem(machine_states=["Off" for _ in range(number_machines)],
                                            machine_control_actions=["K" for _ in range(number_machines)],
                                            buffer_states=[0 for _ in range(number_machines-1)],
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
                                                           ))
                 ):
        #the ManufacturingSystem is with updated machine and microgrid states S_{t+1}
        #from these we obtain the set of all admissible microgrid actions for adjusting the status of [solar, wind, generator], 
        #and the set of all admissible microgrid actions for energy purchased/discharged
        self.System=System
    
    def List_AdjustingStatus(self):
        #return all possible microgrid actions for adjusting the status [solar, wind, generator]#
        microgrid_action_set_list_adjustingstatus=[]
        for adjust_solar in range(2):
            for adjust_wind in range(2):
                for adjust_generator in range(2):
                    microgrid_action_set_list_adjustingstatus.append([adjust_solar, adjust_wind, adjust_generator])
        return microgrid_action_set_list_adjustingstatus

    def List_PurchasedDischarged(self, 
                                 actions_solar=[0,0,0],
                                 actions_wind=[0,0,0],
                                 actions_generator=[0,0,0]):
        #return all possible microgrid actions for the use of the purchased energy and the energy discharge#
        #actions_solar, actions_wind, actions_generator are the actions to be taken at current system states#
        TotalSoldBack=actions_solar[3-1]+actions_wind[3-1]+actions_generator[3-1]
        #Total amount of sold back energy#
        TotalBattery=actions_solar[2-1]+actions_wind[2-1]+actions_generator[2-1]
        #Total amount if energy charged to the battery#
        self.SOC_min = self.System.grid.SOC_min  # Muhammad
        SOC_Condition=self.System.grid.SOC-rate_battery_discharge*Delta_t/charging_discharging_efficiency-self.SOC_min
        #The condition for SOC at the current system state#
        E_mfg=0
        for i in range(number_machines):
            E_mfg=E_mfg+self.System.machine[i].EnergyConsumption()
        #total energy consumed by the manufacturing system, summing over all machines#
        p_hat=E_mfg-(actions_solar[1-1]+actions_wind[1-1]+actions_generator[1-1])
        if p_hat<0:
            p_hat=0
        #Set the p_hat#
        p_tilde=E_mfg-(actions_solar[1-1]+actions_wind[1-1]+actions_generator[1-1]+rate_battery_discharge*Delta_t)
        if p_tilde<0:
            p_tilde=0
        #Set the p_tilde#
        ####Generate the list of the set of all admissible actions_purchased and actions_discharged according to the table in the paper####
        #microgrid_action_set_list_purchased_discharged=[[action_purchased[0], action_purchased[1]], action_discharged]
        microgrid_action_set_list_purchased_discharged=[]
        if TotalSoldBack>0 and TotalBattery>0 and SOC_Condition>0:
            microgrid_action_set_list_purchased_discharged=[ [[0,0], 0] ]
        elif TotalSoldBack>0 and TotalBattery>0 and SOC_Condition<=0:
            microgrid_action_set_list_purchased_discharged=[ [[0,0], 0] ]
        elif TotalSoldBack>0 and TotalBattery<=0 and SOC_Condition>0:
            microgrid_action_set_list_purchased_discharged=[ [[0,0], 0] , [[0,0], rate_battery_discharge*Delta_t] ]
        elif TotalSoldBack>0 and TotalBattery<=0 and SOC_Condition<=0:
            microgrid_action_set_list_purchased_discharged=[ [[0,0], 0] ]
        elif TotalSoldBack<=0 and TotalBattery>0 and SOC_Condition>0:
            microgrid_action_set_list_purchased_discharged=[ [[p_hat, 0], 0] , [[0, p_hat], 0] ]
        elif TotalSoldBack<=0 and TotalBattery>0 and SOC_Condition<=0:
            microgrid_action_set_list_purchased_discharged=[ [[p_hat, 0], 0] , [[0, p_hat], 0] ]
        elif TotalSoldBack<=0 and TotalBattery<=0 and SOC_Condition>0:
            microgrid_action_set_list_purchased_discharged=[ [[p_hat, 0], 0] , [[0, p_hat], 0] , [[p_tilde, 0], rate_battery_discharge*Delta_t] ]
        else:
            microgrid_action_set_list_purchased_discharged=[ [[p_hat, 0], 0] , [[0, p_hat], 0] ]
        #return the list of the set of all admissible actions_purchased and actions_discharged#
        return microgrid_action_set_list_purchased_discharged
    



"""
Generate the set of all admissible machine actions based on the current state S_{t+1} of the manufacturing system.
The set of all machine actions will be stored in a tree with branches 1 or 2, the depth of the tree = num_machines.
Search the tree and return all possible admissible machine actions as a list
"""
class MachineActionTree(object):
    
    def __init__(self, 
                 machine_action):
        self.root=machine_action
        self.left_child=None
        self.right_child=None
        self.machine_action_set_list=[]
    
    def InsertLeft(self, machine_action):
        #insert the left child of the tree from the root#
        if self.left_child == None:
            self.left_child = MachineActionTree(machine_action)
        else:
            new_node = MachineActionTree(machine_action)
            new_node.left_child = self.left_child
            self.left_child = new_node
            
    def InsertRight(self, machine_action):
        #insert the right child of the tree from the root#
        if self.right_child == None:
            self.right_child = MachineActionTree(machine_action)
        else:
            new_node = MachineActionTree(machine_action)
            new_node.right_child = self.right_child
            self.right_child = new_node
        
    def BuildTree(self, System, level, tree):
        #build the tree with root "ROOT", each level corresponding to admissible machine actions for the machine at that level#
        if level < number_machines:
            if System.machine_states[level]==1: #("Opr" or 1)
                tree.InsertLeft("K")
                self.BuildTree(System, level+1, tree.left_child)
                tree.InsertRight("H")
                self.BuildTree(System, level+1, tree.right_child)
            elif System.machine_states[level]== 4: #("Blo" or 4)
                tree.InsertLeft("K")
                self.BuildTree(System, level+1, tree.left_child)
                tree.InsertRight("H")
                self.BuildTree(System, level+1, tree.right_child)
            elif System.machine_states[level]== 3: #("Sta" or 3)
                tree.InsertLeft("K")
                self.BuildTree(System, level+1, tree.left_child)
                tree.InsertRight("H")
                self.BuildTree(System, level+1, tree.right_child)
            elif System.machine_states[level]== 0: #("Off" or 0)
                tree.InsertLeft("K")
                self.BuildTree(System, level+1, tree.left_child)
                tree.InsertRight("W")
                self.BuildTree(System, level+1, tree.right_child)
            else:
                tree.InsertLeft("K")
                self.BuildTree(System, level+1, tree.left_child)
        else:
            return None

    def TraverseTree(self, level, tree, machine_action_list):
        #traverse the tree and output the set of all admissible machine actions as a list#
        if level < number_machines:
            machine_action_list.append(tree.left_child.root)
            self.TraverseTree(level+1, tree.left_child, machine_action_list)
            machine_action_list.pop()
            if tree.right_child == None:
                return None
            else:
                machine_action_list.append(tree.right_child.root)
                self.TraverseTree(level+1, tree.right_child, machine_action_list)
                machine_action_list.pop()
        else:
            machine_action_list_copy=machine_action_list.copy()
            self.machine_action_set_list.append(machine_action_list_copy)
            return None

#initialize the microgrid and manufacturing system
def SystemInitialize(initial_machine_states, initial_machine_actions, initial_buffer_states):
    #the System is initialized with initial machine and buffer states, all other parameters are set to be 0
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
    System=ManufacturingSystem(machine_states=initial_machine_states,
                               machine_control_actions=initial_machine_actions,
                               buffer_states=initial_buffer_states,
                               grid=grid
                               )
    return System
    
    
"""
################################ MAIN TESTING FILE #####################################
################################ FOR DEBUGGING ONLY #####################################

testing on random admissible actions
testing on the generation of admissible actions
"""
if __name__ == "__main__":
    
    #set the initial machine states, machine control actions and buffer states
    initial_machine_states=["Opr" for _ in range(number_machines)]
    initial_machine_actions=["K" for _ in range(number_machines)]
    initial_buffer_states=[2 for _ in range(number_machines-1)]
    
    #initialize the system
    System=SystemInitialize(initial_machine_states, initial_machine_actions, initial_buffer_states)
    
    #initialize the theta
    theta=[0,0,0,0,0,0]
    
    targetoutput=0
    number_iteration=100
    file=open('microgrid_manufacturing_system2.txt', 'w')
    print("\n*********************** RUN THE MICROGRID-MANUFACTURING SYSTEM AT "+str(number_iteration)+" STEPS ***********************", file=file)
    for t in range(number_iteration):
        #current states and actions S_t and A_t are stored in class System#
        print("*********************Time Step", t, "*********************", file=file)
        System.PrintSystem(file, t)
        targetoutput+=int(System.throughput()/unit_reward_production)
        #update the theta#
        theta=projection(np.random.uniform(-1,1,size=6))
        #calculate the next states and actions, S_{t+1}, A_{t+1}#       
        next_machine_states, next_buffer_states=System.transition_manufacturing()
        next_workingstatus, next_SOC=System.grid.transition(t)
        print("Battery_degradation_cost: ", System.grid.get_battery_degradation_cost(t))
        next_action=ActionSimulation(System=ManufacturingSystem(machine_states=next_machine_states,
                                                                machine_control_actions=["K" for _ in range(number_machines)],
                                                                buffer_states=next_buffer_states,
                                                                grid=Microgrid(workingstatus=next_workingstatus,
                                                                               SOC=next_SOC,
                                                                               actions_adjustingstatus=[0,0,0],
                                                                               actions_solar=[0,0,0],
                                                                               actions_wind=[0,0,0],
                                                                               actions_generator=[0,0,0],
                                                                               actions_purchased=[0,0],
                                                                               actions_discharged=0,
                                                                               solarirradiance=solarirradiance[t//8640],
                                                                               windspeed=windspeed[t//8640],
                                                                               )
                                                                )
                                    )
        next_actions_adjustingstatus=next_action.MicroGridActions_adjustingstatus()
        next_actions_solar, next_actions_wind, next_actions_generator=next_action.MicroGridActions_SolarWindGenerator(theta)
        next_actions_purchased, next_actions_discharged=next_action.MicroGridActions_PurchasedDischarged(next_actions_solar,
                                                                                                         next_actions_wind,
                                                                                                         next_actions_generator)
        next_machine_control_actions=next_action.MachineActions()
        # actual_capacity=System.grid.get_actual_capacity_Eat(t)
        # print("Battery_degradation_cost: ", System.grid.get_battery_degradation_cost(t))
        grid=Microgrid(workingstatus=next_workingstatus,
                       SOC=next_SOC,
                       actions_adjustingstatus=next_actions_adjustingstatus,
                       actions_solar=next_actions_solar,
                       actions_wind=next_actions_wind,
                       actions_generator=next_actions_generator,
                       actions_purchased=next_actions_purchased,
                       actions_discharged=next_actions_discharged,
                       solarirradiance=solarirradiance[t//8640],
                       windspeed=windspeed[t//8640]
                       )
        System=ManufacturingSystem(machine_states=next_machine_states, 
                                   machine_control_actions=next_machine_control_actions, 
                                   buffer_states=next_buffer_states,
                                   grid=grid
                                   )  
    print("Target Output = ", targetoutput, file=file)
    
    #test the tree structure in the generation of all admissible machine actions#
    #test the generation of all admissible microgrid adjusting actions and actions for energy purchased/discharged#
    print("\n*********************** Test the Machine and Microgrid Action Generation ***********************", file=file)
    #first print the current system parameters#
    System.PrintSystem(file, t)
    #generate the admissible machine actions from the tree structure#
    machine_action_tree=MachineActionTree(machine_action="ROOT")
    machine_action_tree.BuildTree(System, level=0, tree=machine_action_tree)
    machine_action_list=[]
    machine_action_tree.TraverseTree(level=0, tree=machine_action_tree, machine_action_list=[])
    machine_action_set_list=machine_action_tree.machine_action_set_list
    i=1
    for machine_action_list in machine_action_set_list:
        print("admissible machine action", i, "=", machine_action_list, file=file)
        i=i+1
    #generate the admissible microgrid actions for adjusting status and purchased/discharged
    microgrid_action_set_DR=MicrogridActionSet_Discrete_Remainder(System)
    microgrid_action_set_list_adjustingstatus=microgrid_action_set_DR.List_AdjustingStatus()
    i=1
    print("\n", file=file)
    for microgrid_action_list_adjustingstatus in microgrid_action_set_list_adjustingstatus:
        print("admissible microgrid action", i," for adjusting status=", microgrid_action_list_adjustingstatus, file=file)
        i=i+1

    microgrid_action_set_list_purchased_discharged=microgrid_action_set_DR.List_PurchasedDischarged(actions_solar=[0,0,0],
                                                                                                    actions_wind=[0,0,0],
                                                                                                    actions_generator=[0,0,0])
    i=1
    print("\n",file=file)
    for microgrid_action_list_purchased_discharged in microgrid_action_set_list_purchased_discharged:
        print("admissible microgrid action", i," for purchase=", microgrid_action_list_purchased_discharged[0],
              ", admissible microgrid action", i," for discharge=", microgrid_action_list_purchased_discharged[1], file=file)
        i=i+1
        
    file.close()
