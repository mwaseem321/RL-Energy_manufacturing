import numpy as np
import torch as T
import torch.nn.functional as F
from networks import DiscreteActorNetwork, DiscreteCriticNetwork, ContinuousActorNetwork, ContinuousCriticNetwork

class DiscreteAgent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.01, beta=0.01, fc1=64,
                    fc2=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        # print("disc_actor_dims:", actor_dims)
        self.actor = DiscreteActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')
        self.critic = DiscreteCriticNetwork(beta, critic_dims,
                            fc1, fc2, n_agents, n_actions,
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        self.target_actor = DiscreteActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir,
                                        name=self.agent_name+'_target_actor')
        self.target_critic = DiscreteCriticNetwork(beta, critic_dims,
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        # print("observation in choose action: ", observation)
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        # print("disc_state: ", state)
        actions = self.actor.forward(state.clone())
        noise = T.rand(self.n_actions).to(self.actor.device)
        action = actions + noise

        return action.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

class ContinuousAgent:
    def __init__(self, actor_dims, critic_dims, n_actions,
                 n_agents, agent_idx, chkpt_dir, min_action,
                 max_action, alpha=1e-4, beta=1e-3, fc1=64,
                 fc2=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        agent_name = 'agent_%s' % agent_idx
        self.agent_idx = agent_idx
        self.min_action = min_action
        self.max_action = max_action
        # print("cont_actor_dims:", actor_dims)
        self.actor = ContinuousActorNetwork(alpha, actor_dims,n_actions, fc1, fc2)
        self.target_actor = ContinuousActorNetwork(alpha, actor_dims, n_actions, fc1, fc2)

        self.critic = ContinuousCriticNetwork(beta, critic_dims, fc1, fc2,
                                    chkpt_dir=chkpt_dir)
        self.target_critic = ContinuousCriticNetwork(beta, critic_dims, fc1, fc2,
                                           chkpt_dir=chkpt_dir)

        self.update_network_parameters(tau=1)


    def choose_action(self, observation, evaluate=False):
        state = T.tensor(observation, dtype=T.float, device=self.actor.device)
        actions = self.actor.forward(state)
        # print("actions from the actor network in cont_agent: ", actions)

        # Ensure actions is a numpy array
        actions = actions.cpu().detach().numpy()

        # Round first value to 0 or 1
        actions[0][0] = 1 if actions[0][0] > 0.5 else 0

        # Adjust the last three values to make their sum equal to 1
        last_three_sum = np.sum(actions[0][1:])
        if last_three_sum != 0:
            actions[0][1:] = actions[0][1:] / last_three_sum
        else:
            actions[0][1:] = [1 / 3, 1 / 3, 1 / 3]
        # print("Final array: ", actions)
        return actions


    def update_network_parameters(self, tau=None):
        tau = tau or self.tau

        src = self.actor
        dest = self.target_actor

        for param, target in zip(src.parameters(), dest.parameters()):
            # print("param.data_size: ", param.data.shape, "target data: ", target.data.shape)
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

        src = self.critic
        dest = self.target_critic

        for param, target in zip(src.parameters(), dest.parameters()):
            # print("param.data_size: ",param.data.shape, "target data: ",target.data.shape )
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self, memory, agent_list):
        if not memory.ready():
            return

        actor_states, states, actions, rewards,\
            actor_new_states, states_, dones = memory.sample_buffer()

        device = self.actor.device

        states = T.tensor(np.array(states), dtype=T.float, device=device)
        rewards = T.tensor(np.array(rewards), dtype=T.float, device=device)
        states_ = T.tensor(np.array(states_), dtype=T.float, device=device)
        dones = T.tensor(np.array(dones), device=device)

        actor_states = [T.tensor(actor_states[idx],
                                 device=device, dtype=T.float)
                        for idx in range(len(agent_list))]
        actor_new_states = [T.tensor(actor_new_states[idx],
                                     device=device, dtype=T.float)
                            for idx in range(len(agent_list))]
        actions = [T.tensor(actions[idx], device=device, dtype=T.float)
                   for idx in range(len(agent_list))]

        with T.no_grad():
            new_actions = T.cat([agent.target_actor(actor_new_states[idx])
                                 for idx, agent in enumerate(agent_list)],
                                dim=1)
            critic_value_ = self.target_critic.forward(
                                states_, new_actions).squeeze()
            critic_value_[dones[:, self.agent_idx]] = 0.0
            target = rewards[:, self.agent_idx] + self.gamma * critic_value_

        old_actions = T.cat([actions[idx] for idx in range(len(agent_list))],
                            dim=1)
        critic_value = self.critic.forward(states, old_actions).squeeze()
        critic_loss = F.mse_loss(target, critic_value)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic.optimizer.step()

        actions[self.agent_idx] = self.actor.forward(
                actor_states[self.agent_idx])
        actions = T.cat([actions[i] for i in range(len(agent_list))], dim=1)
        actor_loss = -self.critic.forward(states, actions).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor.optimizer.step()

        self.update_network_parameters()