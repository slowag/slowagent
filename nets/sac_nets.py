import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from collections import deque


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


class ActorConcat(Actor):
    def __init__(self, env, args):
        nn.Module.__init__(self)
        self.args = args
        hidden_dim = args.actor_hidden_dim
        self.blocks = nn.ModuleList()
        self.input_dim = np.array(env.single_observation_space.shape).prod()
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

    def forward(self, obs):
        out = obs
        for i, block in enumerate(self.blocks):
            if i != 0:
                out = torch.cat([obs, out], dim=1)
            out = F.relu(block(out))
        last_hidden = torch.cat([obs, out], dim=1)
        mean = self.fc_mean(last_hidden)
        log_std = self.fc_logstd(last_hidden)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std


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

    def get_action(self, x, hidden_activations):
        mean, log_std, hidden_activations = self(x, hidden_activations)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        if self.args.trainer == 'delayed_sampled':
            log_prob = log_prob.sum(1, keepdim=True)
            return action, log_prob, torch.exp(log_prob), hidden_activations
        else:
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            return action, log_prob, mean, hidden_activations

    def init_activation(self, x):
        if self.args.get_instant_activations:
            return x
        else:
            return torch.zeros_like(x)

    def get_activations(self, x):
        hidden_activations = []
        for block in self.blocks:
            x = F.relu(block(x))
            hidden_activations.append(self.init_activation(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        hidden_activations.append((self.init_activation(mean), self.init_activation(log_std)))

        return hidden_activations

    def learn_action(self, obs):
        hidden_activations = self.get_activations(obs[0])
        for i, ob in enumerate(obs):
            action, log_prob, mean, hidden_activations = self.get_action(ob, hidden_activations)
        return action, log_prob, mean


class ActorSlowLinear(ActorSlow):
    def __init__(self, env, args):
        nn.Module.__init__(self)
        bias = False
        self.args = args
        self.input_dim = np.array(env.single_observation_space.shape).prod()
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
        self.last_obs = deque(maxlen=3)

    def forward(self, obs, hidden_activations):
        if self.training:
            self.last_obs.append(obs)
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


class ActorSlowInParallel(ActorSlowLinear):
    def __init__(self, env, args):
        nn.Module.__init__(self)
        bias = True
        self.args = args
        self.hidden_dim = args.actor_hidden_dim
        self.overall_hidden_dim = self.hidden_dim * (args.N_hidden_layers + 1)
        self.obs_dim = np.array(env.single_observation_space.shape).prod()
        self.action_dim = np.prod(env.single_action_space.shape)

        print('self.obs_dim', self.obs_dim)
        print('self.action_dim', self.action_dim)
        print('self.overall_hidden_dim', self.overall_hidden_dim)

        self.input_dim = self.obs_dim + self.overall_hidden_dim
        self.output_dim = self.overall_hidden_dim + 2*self.action_dim

        self.in_dim = [0] + [self.obs_dim] + [self.obs_dim + self.hidden_dim * i for i in range(1, args.N_hidden_layers + 1 + 1)]
        self.out_dim = [0] + [self.hidden_dim * i for i in range(1, args.N_hidden_layers + 1 + 1)] + [self.output_dim]

        print('self.in_dim ', self.in_dim)
        print('self.out_dim ', self.out_dim)

        self.register_buffer('mask', torch.zeros(self.output_dim, self.input_dim))

        self.fc_mean = self.layer_init(nn.Linear(self.input_dim, self.output_dim , bias=bias))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high[0] - env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high[0] + env.action_space.low[0]) / 2.0, dtype=torch.float32)
        )
        self.last_obs = deque(maxlen=3)

    def layer_init(self, layer, bias_const=0.0):
        layer.weight.data = torch.zeros_like(layer.weight.data)
        for i in range(self.args.N_hidden_layers+1+1):
            nn.init.kaiming_normal_(layer.weight[self.out_dim[i]:self.out_dim[i+1], self.in_dim[i]:self.in_dim[i+1]])
            self.mask[self.out_dim[i]:self.out_dim[i+1], self.in_dim[i]:self.in_dim[i+1]] = 1
        # if layer.bias:
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, obs, hidden_activations):
        new_hidden_activations = []
        x = torch.cat([obs, hidden_activations[0]], dim=1)
        self.fc_mean.weight.data = self.fc_mean.weight.data * self.mask
        new_x = self.fc_mean(x)
        new_hidden_activations.append(F.relu(new_x[:, :-2*self.action_dim]))
        new_mean = new_x[:, -2*self.action_dim:-self.action_dim].clone()
        new_log_std = new_x[:, -self.action_dim:].clone()
        new_hidden_activations.append((new_mean, new_log_std))
        mean, log_std = hidden_activations[-1]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std, new_hidden_activations

    def get_activations(self, x, last_action=None, last_reward=None):
        hidden_activations = []
        hidden_activations.append(torch.zeros(x.shape[0], self.overall_hidden_dim).to(x.device))
        hidden_activations.append((torch.zeros(x.shape[0], self.action_dim).to(x.device),
                                   torch.zeros(x.shape[0], self.action_dim).to(x.device)))
        return hidden_activations


class ActorAllForwarZerodInParallel(ActorSlowInParallel):
    def layer_init(self, layer, bias_const=0.0):
        layer.weight.data = torch.zeros_like(layer.weight.data)
        for i in range(self.args.N_hidden_layers+1+1):
            nn.init.kaiming_normal_(layer.weight[self.out_dim[i]:self.out_dim[i+1], self.in_dim[i]:self.in_dim[i+1]])
            self.mask[self.out_dim[i]:self.out_dim[-1], self.in_dim[i]:self.in_dim[i+1]] = 1

        torch.nn.init.constant_(layer.bias, bias_const)
        return layer


class ActorSlowSkip(ActorSlow):
    def __init__(self, env, args):
        nn.Module.__init__(self)
        self.args = args
        hidden_dim = args.actor_hidden_dim
        self.blocks = nn.ModuleList()
        self.input_dim = np.array(env.single_observation_space.shape).prod()
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
        self.args.concat_obs = False
        hidden_dim = args.actor_hidden_dim
        self.blocks = nn.ModuleList()
        self.input_dim = np.array(env.single_observation_space.shape).prod()
        # self.blocks.append(nn.LSTMCell(self.input_dim, hidden_dim))
        self.blocks.append(nn.Linear(self.input_dim, hidden_dim))
        for _ in range(args.N_hidden_layers):
            self.blocks.append(nn.LSTMCell(hidden_dim, hidden_dim))
        self.fc_mean = nn.Linear(hidden_dim, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(hidden_dim, np.prod(env.single_action_space.shape))
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
                input = input[0]
                h, c = block(input, hs)
            new_hidden_activations.append((h, c))

        input = hidden_activations[-2][0]
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


class ActorSlowLSTMEfficient(ActorSlowLSTM):
    def forward(self, obs, hidden_activations):
        new_hidden_activations = []
        for input, hs, block in zip([(obs, None)] + hidden_activations[:-2], hidden_activations[:-1], self.blocks):
            if isinstance(block, nn.Linear):
                h = F.relu(F.linear(input[0], block.weight.clone(), block.bias.clone()))
                c = None
            else:
                input = input[0]
                h, c = torch.nn.modules.rnn._VF.lstm_cell(input, hs,
                                                          block.weight_ih.clone(),
                                                          block.weight_hh.clone(),
                                                          block.bias_ih.clone(),
                                                          block.bias_hh.clone())
            new_hidden_activations.append((h, c))

        input = hidden_activations[-2][0]
        new_mean = F.linear(input, self.fc_mean.weight.clone(), self.fc_mean.bias.clone())
        new_log_std = F.linear(input, self.fc_logstd.weight.clone(), self.fc_logstd.bias.clone())
        new_hidden_activations.append((new_mean, new_log_std))
        mean, log_std = hidden_activations[-1]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, new_hidden_activations

    def get_activations(self, x, last_action=None, last_reward=None):
        hidden_activations = []
        for block in self.blocks:
            if isinstance(block, nn.Linear):
                h = F.relu(F.linear(x, block.weight.clone(), block.bias.clone()))
                c = None
            else:
                lstm_state = (
                    torch.zeros(x.shape[0], block.hidden_size).to(x.device),
                    torch.zeros(x.shape[0], block.hidden_size).to(x.device),
                )
                h, c = torch.nn.modules.rnn._VF.lstm_cell(h, lstm_state,
                                                          block.weight_ih.clone(),
                                                          block.weight_hh.clone(),
                                                          block.bias_ih.clone(),
                                                          block.bias_hh.clone())
                h, c = self.init_activation(h), self.init_activation(c)

            hidden_activations.append((h, c))
            x = h

        mean = F.linear(h, self.fc_mean.weight.clone(), self.fc_mean.bias.clone())
        log_std = F.linear(h, self.fc_logstd.weight.clone(), self.fc_logstd.bias.clone())
        hidden_activations.append((self.init_activation(mean), self.init_activation(log_std)))

        return hidden_activations