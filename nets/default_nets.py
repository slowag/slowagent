import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        bias = True
        self.args = args
        hidden_dim = args.actor_hidden_dim
        self.blocks = nn.ModuleList()
        self.input_dim = np.array(env.single_observation_space.shape).prod()
        if args.add_last_action:
            self.input_dim += np.prod(env.single_action_space.shape)
        if args.add_last_reward:
            self.input_dim += 1
        self.blocks.append(nn.Linear(self.input_dim, hidden_dim))
        for _ in range(args.N_hidden_layers):
            self.blocks.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_mean = nn.Linear(hidden_dim, np.prod(env.single_action_space.shape), bias=bias)
        self.fc_logstd = nn.Linear(hidden_dim, np.prod(env.single_action_space.shape), bias=bias)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high[0] - env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high[0] + env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        for block in self.blocks:
            x = F.relu(block(x))
        if self.args.train_only_last_layer:
            x = x.detach()
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def sample_action(self, mean, log_std):
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_action(self, x):
        mean, log_std = self(x)
        action, log_prob, mean = self.sample_action(mean, log_std)
        return action, log_prob, mean

    def backward(self, *args):
        pass


class ActorSkip(Actor):
    def __init__(self, env, args):
        nn.Module.__init__(self)
        self.args = args
        hidden_dim = args.actor_hidden_dim
        self.blocks = nn.ModuleList()
        self.input_dim = np.array(env.single_observation_space.shape).prod()
        if args.add_last_action:
            self.input_dim += np.prod(env.single_action_space.shape)
        if args.add_last_reward:
            self.input_dim += 1
        self.blocks.append(nn.Linear(self.input_dim, hidden_dim))
        for _ in range(args.N_hidden_layers):
            self.blocks.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_mean = nn.Linear(self.input_dim + (args.N_hidden_layers+1)*hidden_dim, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(self.input_dim + (args.N_hidden_layers+1)*hidden_dim, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high[0] - env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high[0] + env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )

    def forward(self, obs):
        hidden_activations = []
        out = obs
        for block in self.blocks:
            out = F.relu(block(out))
            hidden_activations.append(out)

        last_hidden = torch.cat([obs] + hidden_activations, dim=1)
        mean = self.fc_mean(last_hidden)
        log_std = self.fc_logstd(last_hidden)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std


class ActorResSkip(ActorSkip):
    def forward(self, obs):
        hidden_activations = []
        res = 0
        input = obs
        for block in self.blocks:
            out = F.relu(block(input))
            input = out + res
            hidden_activations.append(input)
            res = out

        last_hidden = torch.cat([obs] + hidden_activations, dim=1)
        mean = self.fc_mean(last_hidden)
        log_std = self.fc_logstd(last_hidden)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std


class PPOAgentMLP(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64, bias=False),
            nn.Tanh(),
            nn.Linear(64, 64, bias=False),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )
        self.actor = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64, bias=False),
            nn.Tanh(),
            nn.Linear(64, 64, bias=False),
            nn.Tanh(),
            nn.Linear(64, envs.single_action_space.n),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)