import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
class DiscreteCriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                    n_agents, n_actions, name, chkpt_dir):
        super(DiscreteCriticNetwork, self).__init__()
        # print("input_dims: ",input_dims, "n_actions: ", n_actions, "n_agents: ", n_agents )
        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims) #input_dims+n_agents*n_actions
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class DiscreteActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir, printVersions = True):
        super(DiscreteActorNetwork, self).__init__()
        T.autograd.set_detect_anomaly(True)
        self.printVersions = printVersions
        self.chkpt_file = os.path.join(chkpt_dir, name)
        # print("input_dims", input_dims)
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.to(self.device)

    def forward(self, state):
        # x = T.nn.functional.linear(state, self.fc1.weight.clone(), self.fc1.bias)
        # x1 = F.relu(T.nn.functional.linear(x, self.fc2.weight.clone(), self.fc2.bias))
        # # pi = T.softmax(self.pi(x1), dim=1)
        # pi = T.softmax(T.nn.functional.linear(x1, self.pi.weight.clone(), self.pi.bias), dim=1)
        x = self.fc1(state) #F.relu(self.fc1(state))
        x1= F.relu(x)
        x2 = self.fc2(x1) #F.relu(self.fc2(x))
        x3= self.pi(x2).detach()
        pi = T.softmax(x3, dim=1) #T.softmax(self.pi(x1), dim=1)
        # if self.printVersions:
        #     print('versions:')
        #     print('x:', x._version)
        #     print('x1:', x._version)
        #     print('pi:', pi._version)
        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

class ContinuousActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, fc1_dims=64, fc2_dims=64, chkpt_dir='models/'):
        super(ContinuousActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir,
                                            'actor_continuous')
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.alpha = nn.Linear(fc2_dims, n_actions)
        self.beta = nn.Linear(fc2_dims, n_actions)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)

    # def forward(self, state):
    #     if len(state.size()) == 1:
    #         state = state.unsqueeze(0)
    #
    #     x = F.relu(self.fc1(state))
    #     x = F.relu(self.fc2(x))
    #
    #     alpha = F.relu(self.alpha(x)) + 1.0
    #     alpha = T.mean(alpha).view(1, 1)
    #
    #     beta = F.relu(self.beta(x)) + 1.0
    #     beta = T.mean(beta).view(1, 1)
    #
    #     dist = Beta(alpha, beta)
    #     action1 = dist.sample()
    #     action2 = dist.sample()
    #     print("action1: ", action1, "action2: ", action2)
    #     return action1, action2

    def forward(self, state):
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # print("x.shape", x.shape)
        # Flatten the batch dimension for alpha and beta layers
        # x = x.view(-1, x.size(1))

        alpha = F.relu(self.alpha(x)) + 1.0
        if alpha.size() == T.Size([5, 64]):
            # Reduce the size to [5, 1] by taking the mean along the second dimension
            alpha = T.mean(alpha, dim=1, keepdim=True)
        else:
            alpha = alpha

        # if alpha.size() == T.Size([5, 1]):
        #     # Reduce the size to [1, 1] by taking the mean and then reducing to a scalar
        #     alpha_mean = T.mean(alpha)
        #     alpha = alpha_mean.item()  # Convert tensor to scalar
        # else:
        #     # Handle other cases if necessary
        #     alpha = alpha
        # print("alpha: ", alpha)
        beta = F.relu(self.beta(x)) + 1.0
        if beta.size() == T.Size([5, 64]):
            # Reduce the size to [5, 1] by taking the mean along the second dimension
            beta = T.mean(beta, dim=1, keepdim=True)
        else:
            beta = beta

        # if beta.size() == T.Size([5, 1]):
        #     # Reduce the size to [1, 1] by taking the mean and then reducing to a scalar
        #     beta_mean = T.mean(beta)
        #     beta = beta_mean.item()  # Convert tensor to scalar
        # else:
        #     # Handle other cases if necessary
        #     beta = beta
        # print("beta: ",beta)

        dist = Beta(alpha, beta)
        action1 = dist.sample()
        action2 = dist.sample()
        # print("action1: ", action1, "action2: ", action2)
        return action1, action2

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ContinuousCriticNetwork(nn.Module):
    def __init__(self, alpha, input_dims,
                 fc1_dims=64, fc2_dims=64, chkpt_dir='models/'):
        super(ContinuousCriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir,
                                            'critic_continuous')
        # print("input dims: ", input_dims)
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        # print("state and action:", state.shape, action.shape)
        x = T.tanh(self.fc1(T.cat([state, action], dim=1)))
        x = T.tanh(self.fc2(x))
        v = self.v(x)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))