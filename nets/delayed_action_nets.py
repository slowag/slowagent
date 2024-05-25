import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .default_nets import Actor


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class ActorSlow(Actor):
    def forward(self, obs, hidden_activations):
        new_hidden_activations = []
        for input, block in zip([obs] + hidden_activations[:-2], self.blocks):
            out = F.relu(block(input))
            new_hidden_activations.append(out)
        new_mean = self.fc_mean(hidden_activations[-2])
        new_log_std = self.fc_logstd(hidden_activations[-2])
        new_hidden_activations.append((new_mean, new_log_std))
        mean, log_std = hidden_activations[-1]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, new_hidden_activations

    def get_action(self, x, hidden_activations, last_reward=None, last_action=None):
        mean, log_std, hidden_activations = self(x, hidden_activations)
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
        return action, log_prob, mean, hidden_activations

    def init_activation(self, x):
        if self.args.get_instant_activations:
            return x
        else:
            return torch.zeros_like(x)

    def get_activations(self, x, last_action=None, last_reward=None):
        hidden_activations = []
        for block in self.blocks:
            x = F.relu(block(x))
            hidden_activations.append(self.init_activation(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        hidden_activations.append((self.init_activation(mean), self.init_activation(log_std)))

        return hidden_activations

    def learn_action(self, obs, last_actions=None):
        hidden_activations = self.get_activations(obs[0])
        for i, ob in enumerate(obs):
            action, log_prob, mean, hidden_activations = self.get_action(ob, hidden_activations)
        return action, log_prob, mean


class ActorSlowLinear(ActorSlow):
    def __init__(self, env, args):
        nn.Module.__init__(self)
        bias = True
        self.args = args
        self.input_dim = np.array(env.single_observation_space.shape).prod()
        if args.add_last_action:
            self.input_dim += np.prod(env.single_action_space.shape)
        if args.add_last_reward:
            self.input_dim += 1
        self.fc_mean = nn.Linear(self.input_dim, np.prod(env.single_action_space.shape), bias=bias)
        self.fc_logstd = nn.Linear(self.input_dim, np.prod(env.single_action_space.shape), bias=bias)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high[0] - env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high[0] + env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )

    def forward(self, obs, hidden_activations):
        new_hidden_activations = []
        new_mean = self.fc_mean(obs)
        new_log_std = self.fc_logstd(obs)
        new_hidden_activations.append((new_mean, new_log_std))
        mean, log_std = hidden_activations[-1]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std, new_hidden_activations

    def get_activations(self, x, last_action=None, last_reward=None):
        hidden_activations = []
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        hidden_activations.append((self.init_activation(mean), self.init_activation(log_std)))
        return hidden_activations


class ActorSlowSkip(ActorSlow):
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

    def forward(self, obs, hidden_activations):
        new_hidden_activations = []
        for input, block in zip([obs] + hidden_activations[:-2], self.blocks):
            out = F.relu(block(input))
            new_hidden_activations.append(out)
        cat_hidden = torch.cat([obs] + hidden_activations[:-1], dim=1)
        new_mean = self.fc_mean(cat_hidden)
        new_log_std = self.fc_logstd(cat_hidden)
        new_hidden_activations.append((new_mean, new_log_std))
        mean, log_std = hidden_activations[-1]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, new_hidden_activations

    def get_activations(self, x,  last_action=None, last_reward=None):
        obs = x
        hidden_activations = []
        for block in self.blocks:
            x = F.relu(block(x))
            hidden_activations.append(self.init_activation(x))

        x = torch.cat([obs] + hidden_activations, dim=1)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        hidden_activations.append((self.init_activation(mean), self.init_activation(log_std)))

        return hidden_activations


class ActorSlowConcat(ActorSlow):
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
            self.blocks.append(nn.Linear(self.input_dim + hidden_dim, hidden_dim))
        self.fc_mean = nn.Linear(self.input_dim + hidden_dim, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(self.input_dim + hidden_dim, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high[0] - env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high[0] + env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )

    def forward(self, obs, hidden_activations):
        new_hidden_activations = []
        out = F.relu(self.blocks[0](obs))
        new_hidden_activations.append(out)
        for input, block in zip(hidden_activations[:-2], self.blocks[1:]):
            input = torch.cat([obs, input], dim=1)
            out = F.relu(block(input))
            new_hidden_activations.append(out)
        input = torch.cat([obs, hidden_activations[-2]], dim=1)
        new_mean = self.fc_mean(input)
        new_log_std = self.fc_logstd(input)
        new_hidden_activations.append((new_mean, new_log_std))
        mean, log_std = hidden_activations[-1]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, new_hidden_activations

    def get_activations(self, x,  last_action=None, last_reward=None):
        obs = x
        hidden_activations = []
        x = F.relu(self.blocks[0](x))
        hidden_activations.append(self.init_activation(x))
        for block in self.blocks[1:]:
            x = torch.cat([obs, x], dim=1)
            x = F.relu(block(x))
            hidden_activations.append(self.init_activation(x))

        x = torch.cat([obs, x], dim=1)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        hidden_activations.append((self.init_activation(mean), self.init_activation(log_std)))

        return hidden_activations


class ActorSlowResSkip(ActorSlowSkip):
    def forward(self, obs, hidden_activations):
        new_hidden_activations = []
        res = 0
        for input, block in zip([obs] + hidden_activations[:-2], self.blocks):
            out = F.relu(block(input))
            new_hidden_activations.append((out+res)/np.sqrt(2))
            res = out
        cat_hidden = torch.cat([obs] + hidden_activations[:-1], dim=1)
        new_mean = self.fc_mean(cat_hidden)
        new_log_std = self.fc_logstd(cat_hidden)
        new_hidden_activations.append((new_mean, new_log_std))
        mean, log_std = hidden_activations[-1]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, new_hidden_activations


class ActorSlowLSTM(ActorSlow):
    def __init__(self, env, args):
        nn.Module.__init__(self)
        self.args = args
        self.args.concat_obs = True
        hidden_dim = args.actor_hidden_dim
        self.blocks = nn.ModuleList()
        self.input_dim = np.array(env.single_observation_space.shape).prod()
        if args.add_last_action:
            self.input_dim += np.prod(env.single_action_space.shape)
        if args.add_last_reward:
            self.input_dim += 1

        self.add_input_hidden_dim = self.input_dim if args.concat_obs else 0
        # self.blocks.append(nn.LSTMCell(self.input_dim, hidden_dim))
        self.blocks.append(nn.Linear(self.input_dim, hidden_dim))
        for _ in range(args.N_hidden_layers):
            self.blocks.append(nn.LSTMCell(hidden_dim + self.add_input_hidden_dim, hidden_dim))
        self.fc_mean = nn.Linear(hidden_dim + self.add_input_hidden_dim, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(hidden_dim + self.add_input_hidden_dim, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high[0] - env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high[0] + env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )

    def forward(self, obs, hidden_activations):
        new_hidden_activations = []
        for input, hs, block in zip([(obs, None)] + hidden_activations[:-2], hidden_activations[:-1], self.blocks):
            if isinstance(block, nn.Linear):
                h = F.relu(block(input[0]))
                c = None
            else:
                input = torch.cat([obs, input[0]], dim=1) if self.args.concat_obs else input[0]
                h, c = block(input, hs)
            new_hidden_activations.append((h, c))

        input = torch.cat([obs, hidden_activations[-2][0]], dim=1) if self.args.concat_obs else hidden_activations[-2][0]
        new_mean = self.fc_mean(input)
        new_log_std = self.fc_logstd(input)
        new_hidden_activations.append((new_mean, new_log_std))
        mean, log_std = hidden_activations[-1]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, new_hidden_activations

    def get_activations(self, x, last_action=None, last_reward=None):
        obs = x
        hidden_activations = []
        for block in self.blocks:
            if isinstance(block, nn.Linear):
                h = F.relu(block(x))
                c = None
            else:
                lstm_state = (
                    torch.zeros(x.shape[0], block.hidden_size).to(x.device),
                    torch.zeros(x.shape[0], block.hidden_size).to(x.device),
                )
                x = torch.cat([obs, x], dim=1) if self.args.concat_obs else x
                h, c = block(x, lstm_state)
                h, c = self.init_activation(h), self.init_activation(c)

            hidden_activations.append((h, c))
            x = h

        input = torch.cat([obs, h], dim=1) if self.args.concat_obs else h
        mean = self.fc_mean(input)
        log_std = self.fc_logstd(input)
        hidden_activations.append((self.init_activation(mean), self.init_activation(log_std)))

        return hidden_activations
