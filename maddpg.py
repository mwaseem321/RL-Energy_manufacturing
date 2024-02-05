import numpy as np
import torch as T
import torch.nn.functional as F
from agent import DiscreteAgent, ContinuousAgent
# T.autograd.set_detect_anomaly(True)

class MADDPG:
    def __init__(self, disc_actor_dims,cont_actor_dims, critic_dims, n_agents_disc,n_agents_cont, n_actions_disc, n_actions_cont,
                 alpha=0.01, beta=0.01, fc1=64,
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.discrete_agents = []
        self.continuous_agents = []
        self.n_agents_disc = n_agents_disc
        self.n_agents_cont = n_agents_cont
        self.n_actions_disc = n_actions_disc
        self.n_actions_cont = n_actions_cont
        # print("disc_actor_dims", disc_actor_dims)
        for agent_idx in range(self.n_agents_disc):
            self.discrete_agents.append(DiscreteAgent(disc_actor_dims[agent_idx], critic_dims,
                            n_actions_disc, n_agents_disc+n_agents_cont, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))

        for agent_idx in range(self.n_agents_cont):
            self.continuous_agents.append(ContinuousAgent(cont_actor_dims[agent_idx], critic_dims,
                            n_actions_cont, n_agents_disc+n_agents_cont, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir, min_action= -1, max_action= 1))
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.discrete_agents:
            agent.save_models()
        for agent in self.continuous_agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.discrete_agents:
            agent.load_models()
        for agent in self.continuous_agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        self.agents_all= self.discrete_agents + self.continuous_agents
        # print("self.agents_all", self.agents_all)
        for agent_idx, agent in enumerate(self.agents_all):
            # print("action taken for agent ", agent_idx, agent)
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
            # print("actions after appending: ", actions)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return
        T.autograd.set_detect_anomaly(True)
        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()
        # print("memory sample in learn: ", memory.sample_buffer())
        device = self.agents_all[0].actor.device
        # print("device in learn: ", device)
        # print("action shape in learn: ", actions)
        # print("states type: ", type(states))
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents_all):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states.clone())
            # print("agent_idx: ",agent_idx, "new_pi", new_pi)
            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states.clone())
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions.clone()[agent_idx])

        # print("all_agents_new_actions: ", len(all_agents_new_actions))
        # print("all_agents_new_actions: ", all_agents_new_actions)

        new_actions = []

        for acts in all_agents_new_actions:
            if isinstance(acts, tuple):
                # If acts is a tuple, convert each tensor to the desired device
                concat_new_acts = T.cat(acts, dim=0)
                new_actions.append(concat_new_acts)
            else:
                # If acts is a tensor, convert it to the desired device
                new_actions.append(acts)
        # print("new_actions after concat: ", new_actions)

        for i in range(5, 8):  # Last 3 tensors
            # Extract the first 5 and last 5 values
            first_five = new_actions[i][:5]
            last_five = new_actions[i][-5:]

            # Concatenate the first and last five values along the second dimension
            modified_tensor = T.cat((first_five, last_five), dim=1)

            # Append zeros to each row of the modified tensor
            zeros_tensor = T.zeros((modified_tensor.size(0), 1)).to(device)
            n_modified_tensor = T.cat((modified_tensor, zeros_tensor), dim=1)

            # Update the tensor in the list
            new_actions[i] = n_modified_tensor

        # print("new_actions: ", new_actions)
        # Concatenate the tensors along the specified dimension
        new_actions = T.cat(new_actions, dim=1)
        # print("new_actions after final concat: ", new_actions)

        new_mu_actions = []
        for acts in all_agents_new_mu_actions:
            if isinstance(acts, tuple):
                # If acts is a tuple, convert each tensor to the desired device
                concat_new_acts = T.cat(acts, dim=0)
                new_mu_actions.append(concat_new_acts)
            else:
                # If acts is a tensor, convert it to the desired device
                new_mu_actions.append(acts)
        # print("new_mu_actions after concat: ", new_mu_actions)

        for i in range(5, 8):  # Last 3 tensors
            # Extract the first 5 and last 5 values
            first_five = new_mu_actions[i][:5]
            last_five = new_mu_actions[i][-5:]

            # Concatenate the first and last five values along the second dimension
            modified_tensor = T.cat((first_five, last_five), dim=1)

            # Append zeros to each row of the modified tensor
            zeros_tensor = T.zeros((modified_tensor.size(0), 1)).to(device)
            n_modified_tensor = T.cat((modified_tensor, zeros_tensor), dim=1)

            # Update the tensor in the list
            new_mu_actions[i] = n_modified_tensor

        # print("new_mu_actions: ", new_actions)
        # Concatenate the tensors along the specified dimension
        mu = T.cat(new_mu_actions, dim=1)
        # print("new_actions after final concat: ", new_actions)

        # mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents_all):
            # print(f"agen {agent_idx} is is now in process")
            # print("states_: ", states_.shape, "new_actions:", new_actions.shape)
            critic_value_ = agent.target_critic.forward(states_.clone(), new_actions.clone()).flatten()
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states.clone(), old_actions.clone()).flatten()
            # print("critic_value: ", critic_value.dtype, "critic_value_: ", critic_value_.dtype)
            critic_value_ = critic_value_.double()
            critic_value = critic_value.double()
            target = rewards.clone()[:,agent_idx] + agent.gamma*critic_value_.clone()
            # print("target: ", target.dtype)
            critic_loss = F.mse_loss(target, critic_value)
            # print("critic_loss: ", critic_loss.dtype)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()
            # print("learning finished!")