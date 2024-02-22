import csv

import numpy as np
from matplotlib import pyplot as plt

from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from microgrid_manufacturing_system2 import Microgrid, ManufacturingSystem, ActionSimulation, MicrogridActionSet_Discrete_Remainder, MachineActionTree, SystemInitialize
from projectionSimplex import projection
from env import Env

#set the number of machines
number_machines=5
#set the unit reward of production
unit_reward_production=4000/10000
#the unit reward for each unit of production (10^4$/unit produced), i.e. the r^p, this applies to the end of the machine sequence#

#the discount factor gamma when calculating the total cost#
gamma=0.999

#the seed for reinforcement training initialization of the network weights and biases
seed=2

#the probability of using random actions vs. on-policy optimal actions in each step of training
p_choose_random_action=0.9

import pandas as pd
#read the solar irradiance, wind speed and the rate of consumption charge data from file#
file_SolarIrradiance = "SolarIrradiance.csv"
file_WindSpeed = "WindSpeed.csv"
file_rateConsumptionCharge = "rate_consumption_charge.csv"
#read the solar irradiace
data_solar = pd.read_csv(file_SolarIrradiance)
solarirradiance = np.array(data_solar.iloc[:,3])
#solar irradiance measured by MegaWatt/km^2
#read the windspeed
data_wind = pd.read_csv(file_WindSpeed)
windspeed = np.array(data_wind.iloc[:,3])*3.6
#windspeed measured by km/h=1/3.6 m/s
#read the rate of consumption charge
data_rate_consumption_charge = pd.read_csv(file_rateConsumptionCharge)
rate_consumption_charge = np.array(data_rate_consumption_charge.iloc[:,4])/10
#rate of consumption charge measured by 10^4$/MegaWatt=10 $/kWh



def obs_list_to_state_vector(observation):
    # print("observation in obs_list_to_state_vector: ", observation, "shape: ", len(observation))
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    #scenario = 'simple'
    env = Env()
    agents_disc= 5 # number of machines
    agents_cont= 3 # number of energy sources
    n_agents = agents_disc+agents_cont #env.n, 5 machines and 3 sources of energy
    actor_dims_disc = [5,5,5,5,5]
    actor_dims_cont = [4,4,4]

    critic_dims = sum(actor_dims_disc) + sum(actor_dims_cont) + 32 # 27= 5x3 + 3x4 is the all agents' actions
    # 32 is to accomodate the padded five discrete agents' actions

    # action space is a list of arrays
    n_actions_disc = 3  # each machine has action space of three with H, K and W actions
    n_actions_cont = 4 # continuous action space with a single action at a time
    n_actions= [3,3,3,3,3,4,4,4]
    n_actions_buffer = [4, 4, 4, 4, 4, 4, 4, 4] # used only for buffer initialization
    maddpg_agents = MADDPG(actor_dims_disc,actor_dims_cont, critic_dims, agents_disc, agents_cont,
                           n_actions_disc,n_actions_cont,
                           fc1=128, fc2=128,
                           alpha=0.001, beta=0.01,
                           chkpt_dir='tmp/maddpg/')

    # n_test= [4,4,4,4,4,4,4,4]
    memory = MultiAgentReplayBuffer(1000000, 37, actor_dims_disc+actor_dims_cont,
                        n_actions_buffer, n_agents, batch_size=64)

    PRINT_INTERVAL = 100
    n_episodes = 1000
    episode_length = 2000
    total_steps = 500
    score_history = []
    evaluate = False
    best_score = 0
    # List to store rewards for each episode

    if evaluate:
        maddpg_agents.load_checkpoint()

# Continuous training

# Define your reinforcement learning algorithm and environment

# Initialize variables
    rewards_window = []  # Store last 100 rewards
    rewards_window_discounted= []
    all_rewards = []  # Store all rewards
    all_rewards_discounted = []  # Store all rewards
    episode_count = 0
    converged = False
    half_episodes = n_episodes // 2  # Number of episodes for half threshold #n_episodes
    reward_1 = 0  # based on base code's reward function
    reward_seq = []
    # Training loop
    while not converged:
        obs = env.reset()
        score = 0
        score_discounted= 0
        done = [False] * n_agents
        episode_step = 0

        while not any(done):
            print()
            print("Timestep: ", episode_step)
            print()
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, _, info = env.step(actions, episode_step)
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)
            with open("normal_reward.txt", "a") as file:
                file.write(str(reward) + "\n")

            reward_1 = reward_1 + np.power(gamma, episode_step) * reward  # reward represent the E from main code and reward_1 is the reward from there
            with open("discounted_reward.txt", "a") as file:
                file.write(str(reward_1) + "\n")
            reward_seq.append(reward_1)
            if episode_step >= episode_length:
                done = [True] * n_agents

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 5 == 0 and not evaluate:
                maddpg_agents.learn(memory)
            obs = obs_
            score += reward
            score_discounted+=reward_1
            total_steps += 1
            episode_step += 1

        rewards_window.append(score)
        with open("normal_reward_window.txt", "a") as file:
            file.write(str(score) + "\n")
        all_rewards.append(score)
        rewards_window_discounted.append(score_discounted)
        with open("discounted_reward_window.txt", "a") as file:
            file.write(str(score_discounted) + "\n")
        all_rewards_discounted.append(score_discounted)

        # Check for convergence condition
        if len(all_rewards) >= 100:
            last_100_rewards = rewards_window[-100:]
            variance = np.var(last_100_rewards)
            mean_reward = np.mean(last_100_rewards)
            variance_threshold = 0.1 * mean_reward

            if variance <= variance_threshold and episode_count >= half_episodes:
                converged = True

        # Update episode count
        episode_count += 1

    # Plot rewards
    plt.plot(all_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.savefig('reward_per_episode.png')

    plt.plot(reward_seq)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Reward per Episode')
    plt.savefig('Reward based on previous code.png')

    # testing loop here
    totalcostlist_optimal = [0]
    totalthroughputlist_optimal = [0]
    totalenergydemandlist_optimal = [0]

    # set the total cost, total throughput and the total energy demand#
    totalcost = 0
    totalthroughput = 0
    totalenergydemand = 0
    RL_target_output = 0
    grid = Microgrid(env.working_status,
                     env.SOC,
                     env.actions_adjustingstatus,
                     env.actions_solar,
                     env.actions_wind,
                     env.actions_generator,
                     env.actions_purchased,
                     env.actions_discharged,
                     solarirradiance=solarirradiance[env.t],
                     windspeed=windspeed[env.t],
                     t=env.t
                     )
    system = ManufacturingSystem(env.machine_states, env.machine_control_actions, env.buffer_states, grid)
    obs= env.reset()
    for i in range(100):
        actions = maddpg_agents.choose_action(obs)
        obs_, reward, _, info = env.step(actions, i)
        totalthroughput += system.throughput()
        RL_target_output += int(system.throughput() / unit_reward_production)
        totalthroughputlist_optimal.append(totalthroughput)
        # calculate the total cost at S_t, A_t: E(S_t, A_t)#
        E = system.average_total_cost(rate_consumption_charge[i // 8640])
        # accumulate the total cost#
        totalcost += E
        totalcostlist_optimal.append(totalcost)
        # accumulate the total energy demand#
        totalenergydemand += system.energydemand(rate_consumption_charge[i // 8640])
        totalenergydemandlist_optimal.append(totalenergydemand)
        # determine the next system and grid states#
        env.machine_states, env.buffer_states = system.transition_manufacturing()
        env.working_status, env.SOC = system.grid.transition()
        obs= obs_

    # plot the total throughput, in dollar amount#
    plt.figure(figsize=(14, 10))
    plt.plot([value * 10000 for value in totalthroughputlist_optimal], '-', color='r')
    # plt.plot([value * 10000 for value in totalthroughputlist_benchmark], '--', color='b')
    plt.xlabel('iteration')
    plt.ylabel('total throughput ($)')
    plt.title('Total throughput under optimal policy (red, solid) and benchmark random policy (blue, dashed)')
    plt.savefig('totalthroughput.png')
    plt.show()

    # plot the total throughput, in production units#
    plt.figure(figsize=(14, 10))
    plt.plot([value / unit_reward_production for value in totalthroughputlist_optimal], '-', color='r')
    # plt.plot([value / unit_reward_production for value in totalthroughputlist_benchmark], '--', color='b')
    plt.xlabel('iteration')
    plt.ylabel('total throughput (production unit)')
    plt.title(
        'Total throughput (production unit) under optimal policy (red, solid) and benchmark random policy (blue, dashed)')
    plt.savefig('totalthroughput_unit.png')
    plt.show()

    # plot the total energy demand#
    plt.figure(figsize=(14, 10))
    plt.plot([value * 10000 for value in totalenergydemandlist_optimal], '-', color='r')
    # plt.plot([value * 10000 for value in totalenergydemandlist_benchmark], '--', color='b')
    plt.xlabel('iteration')
    plt.ylabel('total energy cost ($)')
    plt.title('Total energy cost under optimal policy (red, solid) and benchmark random policy (blue, dashed)')
    plt.savefig('totalenergycost.png')
    plt.show()

