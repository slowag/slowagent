import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pickle

class Args:
    N_hidden_layers = 10
    actor_hidden_dim = 256
    action_dim = 8
    obs_dim = 21
    device = 'cuda'
    get_instance_activations = False
    sparse = False
    bias = True
    n_steps = 10000

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class ActorSlowInParallel(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        bias = args.bias
        self.args = args
        print('args.sparse', args.sparse)
        self.hidden_dim = args.actor_hidden_dim
        self.overall_hidden_dim = self.hidden_dim * (args.N_hidden_layers + 1)
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.sparse = args.sparse

        self.in_dim = [0] + [self.obs_dim] + [self.obs_dim + self.hidden_dim * i for i in
                                              range(1, args.N_hidden_layers + 1 + 1)]
        self.out_dim = [0] + [self.hidden_dim * i for i in range(1, args.N_hidden_layers + 1 + 1)] + [
            self.overall_hidden_dim + self.action_dim]

        self.input_dim = self.obs_dim + self.overall_hidden_dim
        self.output_dim = self.overall_hidden_dim + self.action_dim

        self.fc_mean = self.layer_init(nn.Linear(self.input_dim, self.output_dim, bias=bias))
        self.fc_logstd = nn.Linear(self.input_dim, self.action_dim, bias=bias)

        # action rescaling
        self.register_buffer("action_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(1.0, dtype=torch.float32))

        self.weights = self.fc_mean.weight.data.to_sparse().to(args.device)

    def layer_init(self, layer, bias_const=0.0):
        layer.weight.data = torch.zeros_like(layer.weight.data)
        for i in range(self.args.N_hidden_layers + 1 + 1):
            nn.init.kaiming_normal_(
                layer.weight[self.out_dim[i]:self.out_dim[i + 1], self.in_dim[i]:self.in_dim[i + 1]])
        return layer

    def forward(self, obs, hidden_activations):
        new_hidden_activations = []
        x = torch.cat([obs, hidden_activations[0]], dim=1)
        if not self.sparse:
            new_x = self.fc_mean(x)
        else:
            new_x = torch.mm(self.weights, x.squeeze(0).unsqueeze(-1))
            new_x = new_x.squeeze(-1).unsqueeze(0)
        new_hidden_activations.append(F.relu(new_x[:, :-self.action_dim]))
        new_mean = new_x[:, -self.action_dim:]
        new_log_std = self.fc_logstd(x)
        new_hidden_activations.append((new_mean, new_log_std))
        mean, log_std = hidden_activations[-1]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std, new_hidden_activations

    def get_activations(self, x):
        hidden_activations = []
        hidden_activations.append(torch.zeros(x.shape[0], self.overall_hidden_dim).to(x.device))
        hidden_activations.append((torch.zeros(x.shape[0], self.action_dim).to(x.device),
                                   torch.zeros(x.shape[0], self.action_dim).to(x.device)))
        return hidden_activations

    def get_action(self, x, hidden_activations):
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


class ActorSlowInParallelSparse(ActorSlowInParallel):
    def __init__(self, env, args):
        args.sparse = True
        super().__init__(env, args)


class ActorSlow(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        bias = args.bias
        self.args = args
        hidden_dim = args.actor_hidden_dim
        self.blocks = nn.ModuleList()
        self.input_dim = args.obs_dim
        self.blocks.append(nn.Linear(self.input_dim, hidden_dim))
        for _ in range(args.N_hidden_layers):
            self.blocks.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_mean = nn.Linear(hidden_dim, args.action_dim, bias=bias)
        self.fc_logstd = nn.Linear(hidden_dim, args.action_dim, bias=bias)
        # action rescaling
        self.register_buffer("action_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(1.0, dtype=torch.float32))

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
        if self.args.get_instance_activations:
            return x
        else:
            return torch.zeros_like(x)

    def get_activations(self, x, last_action=None, last_reward=None):
        if last_action is not None:
            x = torch.cat([x, last_action], dim=1)
        if last_reward is not None:
            x = torch.cat([x, last_reward.unsqueeze(-1)], dim=1)

        hidden_activations = []
        for block in self.blocks:
            x = F.relu(block(x))
            hidden_activations.append(self.init_activation(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        hidden_activations.append((self.init_activation(mean), self.init_activation(log_std)))

        return hidden_activations

    def learn_action(self, obs, last_actions=None):
        last_action = last_actions[0] if last_actions is not None else None
        hidden_activations = self.get_activations(obs[0], last_action=last_action)

        for i, ob in enumerate(obs):
            last_action = last_actions[i] if last_action is not None else None
            action, log_prob, mean, hidden_activations = self.get_action(ob, hidden_activations,
                                                                         last_action=last_action)

        return action, log_prob, mean


args = Args()

def test_speed(args, actor):
    x = torch.randn(1, args.obs_dim).to(args.device)
    hidden_activations = actor.get_activations(x)

    forward_times = []
    for _ in range(args.n_steps):
        x = torch.randn(1, args.obs_dim).to(args.device)
        forward_time = time.time()
        with torch.no_grad():
            action, _, _, hidden_activations = actor.get_action(x, hidden_activations)
        forward_time = time.time() - forward_time
        forward_times.append(forward_time)

    mean, std, l = np.mean(forward_times), np.std(forward_times), len(forward_times)
    print(f'mean {mean}, std {std}, len {l}')
    return forward_times, mean, std, l

data = {}
for Agent, Name in zip([ActorSlow, ActorSlowInParallel, ActorSlowInParallelSparse],
                       ['ActorSlow', 'ActorSlowInParallel', 'ActorSlowInParallelSparse']):
    data[Name] = []
    n_layers = []
    mean_time = []
    std_time = []
    for i in range(100):
        print(f'N_hidden_layers {i}')
        print('Agent', Name)
        n_layers.append(i)
        args.N_hidden_layers = i
        actor = Agent(None, args).to(args.device)
        print('args', 'N_hidden_layers', args.N_hidden_layers,
              'actor_hidden_dim', args.actor_hidden_dim, 'action_dim', args.action_dim,
              'obs_dim', args.obs_dim, 'get_instance_activations', args.get_instance_activations,
              'device', args.device, 'bias', args.bias)

        _, mean, std, _ = test_speed(args, actor)
        mean_time.append(mean)
        std_time.append(std)

    data[Name].append(n_layers)
    data[Name].append(mean_time)
    data[Name].append(std)

print('data', data)

f = open(f"runs/speedup_{args.device}_steps{args.n_steps}_hd{args.actor_hidden_dim}_b{args.bias}.pkl","wb")
pickle.dump(data, f)