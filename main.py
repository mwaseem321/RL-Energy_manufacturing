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

    # action space is a list of arrays
    n_actions_disc = 3  # each machine has action space of three with H, K and W actions
    n_actions_cont = 4 # continuous action space with a single action at a time
    n_actions= [3,3,3,3,3,4,4,4]
    n_actions_buffer = [4, 4, 4, 4, 4, 4, 4, 4] # used only for buffer initialization
    maddpg_agents = MADDPG(actor_dims_disc,actor_dims_cont, critic_dims, agents_disc, agents_cont,
                           n_actions_disc,n_actions_cont,
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01,
                           chkpt_dir='tmp/maddpg/')

    # n_test= [4,4,4,4,4,4,4,4]
    memory = MultiAgentReplayBuffer(1000000, 37, actor_dims_disc+actor_dims_cont,
                        n_actions_buffer, n_agents, batch_size=5)

    PRINT_INTERVAL = 100
    n_episodes = 50
    episode_length = 100
    total_steps = 500
    score_history = []
    evaluate = False
    best_score = 0
    # List to store rewards for each episode

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(n_episodes):
        obs = env.reset()
        print(f"Episode {i}")
        # print(f"Environment is reset here as {obs}")
        score = 0
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            # print("while loop starts here")
            print()
            print()
            print("Time step: ", episode_step)
            actions = maddpg_agents.choose_action(obs)
            # print(f"Actions chosen from maddpg are {actions}" )
            obs_, reward, _, info = env.step(actions, episode_step)
            # print(f"new obs received at step {episode_step} from step functions is {obs_}")
            # print(f"reward received from step function at timestep {episode_step} is {reward}")
            state = obs_list_to_state_vector(obs)
            # print("state: ", state)
            # print("obs_: ", obs_)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= episode_length:
                done = [True]*n_agents

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 5 == 0 and not evaluate:
                maddpg_agents.learn(memory)
            obs = obs_
            # print("reward: ", reward)
            score += reward
            total_steps += 1
            episode_step += 1
        # print("coming out of while loop!")
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                # maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))

# Compute rolling average of rewards
    window_size = 10
    rolling_avg_rewards = [np.mean(score_history[max(0, j - window_size):j + 1]) for j in range(1, len(score_history))]

    # Plot rolling average rewards
    plt.plot(range(1, len(rolling_avg_rewards) + 1), rolling_avg_rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Rolling Average Reward')
    plt.title('Convergence along Training')
    plt.grid(True)
    plt.show()