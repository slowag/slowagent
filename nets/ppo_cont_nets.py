import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class SlowAgent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.args = args

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            layer_init(nn.Linear(64, 64)),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        l = [layer_init(nn.Linear(64,64)) for _ in range(300)]
        l[0] = layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64))
        self.actor_mean = nn.Sequential(
            *l
        )
        self.actor_head = layer_init(nn.Linear(64,np.prod(envs.single_action_space.shape)))
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_zero_activations(self, x):
        hidden_activations = []
        for block in list(self.actor_mean):
            x = F.tanh(block(x))
            hidden_activations.append(torch.zeros_like(x))
        
        mean = self.actor_head(x)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        hidden_activations.append((torch.zeros_like(mean), std))
        return hidden_activations

    def get_value(self, x):
        for block in list(self.critic):
            x = F.tanh(block(x))
        return x

    def get_action_and_value(self, x, hidden_acts=None, action=None):
        new_hidden = []
        for input, block in zip([x] + hidden_acts[:-2], list(self.actor_mean)):
            out = F.tanh(block(input))
            new_hidden.append(out)

        new_mean = self.actor_head(hidden_acts[-2])
        new_logstd = self.actor_logstd.expand_as(new_mean)
        new_hidden.append((new_mean, torch.exp(new_logstd)))

        action_mean, action_std = hidden_acts[-1]
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x), new_hidden

    def simulate_forward(self, obs, acts):
        hidden_activations = self.get_zero_activations(obs[0])
        for ob,act in zip(obs, acts):
            action, log_prob, entropy, value, hidden_activations = self.get_action_and_value(ob, hidden_activations, act)
        return action, log_prob, entropy, value
    
    def learn_action(self, obs, acts):
        return self.simulate_forward(obs, acts)
    
class SlowSkipAgent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.args = args

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            layer_init(nn.Linear(64, 64)),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            layer_init(nn.Linear(64, 64)),
            layer_init(nn.Linear(64, 64)),
        )
        self.actor_head = layer_init(nn.Linear(64*3 + np.array(envs.single_observation_space.shape).prod() ,np.prod(envs.single_action_space.shape)))
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_zero_activations(self, x):
        hidden_activations = []
        orig_x = x
        for block in list(self.actor_mean):
            x = F.tanh(block(x))
            hidden_activations.append(torch.zeros_like(x))
        
        
        #cat all hidden
        x = torch.cat([orig_x] + hidden_activations, axis=-1)
        
        mean = self.actor_head(x)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        hidden_activations.append((torch.zeros_like(mean), std))
        return hidden_activations

    def get_value(self, x):
        for i,block in enumerate(list(self.critic)):
            if i!=len(list(self.critic))-1:
                x = F.tanh(block(x))
            else:
                x = block(x)
        return x

    def get_action_and_value(self, x, hidden_acts=None, action=None):
        new_hidden = []
        for i, (input, block) in enumerate(zip([x] + hidden_acts[:-2], list(self.actor_mean))):
            if i!=len(list(self.actor_mean))-1:
                out = F.tanh(block(input))
            else:
                out = block(input)
            new_hidden.append(out)

        new_mean = self.actor_head(torch.cat([x] + hidden_acts[:-1], axis=-1))
        new_logstd = self.actor_logstd.expand_as(new_mean)
        new_hidden.append((new_mean, torch.exp(new_logstd)))

        action_mean, action_std = hidden_acts[-1]
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x), new_hidden

    def simulate_forward(self, obs, acts):
        hidden_activations = self.get_zero_activations(obs[0])
        for ob,act in zip(obs, acts):
            action, log_prob, entropy, value, hidden_activations = self.get_action_and_value(ob, hidden_activations, act)
        return action, log_prob, entropy, value
    
    def learn_action(self, obs, acts):
        return self.simulate_forward(obs, acts)