import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, env):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(env.observation_space.shape[0], 128)
        self.action_head = nn.Linear(128,  env.action_space.n)
        self.value_head = nn.Linear(128, 1)
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values

class PolicyDelay(nn.Module):
    def __init__(self, env):
        super(PolicyDelay, self).__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Linear(env.observation_space.shape[0], 128))

        # actor's layer
        self.action_head = nn.Linear(128, env.action_space.n)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def get_init_hidden(self, x):
        hidden_activations = []
        for block in self.blocks:
            x = F.relu(block(x))
            hidden_activations.append(x)
        return hidden_activations

    def forward(self, x, hidden_acts=None):
        """
        forward of both actor and critic
        """
        new_hidden = []
        for input, block in zip([x] + hidden_acts, self.blocks):
            out = F.relu(block(input))
            new_hidden.append(out)

        x = hidden_acts[-1]

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values, new_hidden


class PolicyDelaySkip(nn.Module):

    def __init__(self, env):
        super(PolicyDelaySkip, self).__init__()

        self.blocks = nn.ModuleList()

        self.blocks.append(nn.Linear(env.observation_space.shape[0], 128))

        # actor's layer
        self.action_head = nn.Linear(env.observation_space.shape[0] + 128 * 1, env.action_space.n)

        # critic's layer
        self.value_head = nn.Linear(env.observation_space.shape[0] + 128 * 1, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x, hidden_acts):

        new_hidden = []
        for input, block in zip([x] + hidden_acts, self.blocks):
            out = F.relu(block(input))
            new_hidden.append(out)

        last_hidden = torch.cat([x] + hidden_acts, dim=0)
        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(last_hidden), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(last_hidden)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values, new_hidden

    def get_init_hidden(self, x):
        hidden_activations = []
        for block in self.blocks:
            x = F.relu(block(x))
            hidden_activations.append(x)
        return hidden_activations


class PolicyInd(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, env):
        super(PolicyInd, self).__init__()
        self.actor = nn.Sequential(nn.Linear(env.observation_space.shape[0], 128), nn.LeakyReLU(), nn.Linear(128,  env.action_space.n))
        self.critic = nn.Sequential(nn.Linear(env.observation_space.shape[0], 128), nn.LeakyReLU(), nn.Linear(128, 1))
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        action_prob = F.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_prob, state_values


class PolicyDelayInd(nn.Module):
    def __init__(self, env):
        super(PolicyDelayInd, self).__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(nn.Linear(env.observation_space.shape[0], 128), nn.LeakyReLU()))
        self.blocks.append(nn.Linear(128, env.action_space.n))

        # critic's layer
        self.critic = nn.Sequential(nn.Linear(env.observation_space.shape[0], 128), nn.LeakyReLU(), nn.Linear(128, 1))

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def get_init_hidden(self, x):
        hidden_activations = []
        for block in self.blocks:
            x = block(x)
            hidden_activations.append(x)
        return hidden_activations

    def forward(self, obs, hidden_acts=None):
        new_hidden = []
        for input, block in zip([obs] + hidden_acts, self.blocks):
            out = F.relu(block(input))
            new_hidden.append(out)

        action_prob = F.softmax(hidden_acts[-1], dim=-1)
        state_values = self.critic(obs)

        return action_prob, state_values, new_hidden


class PolicyDelaySkipInd(nn.Module):
    def __init__(self, env):
        super(PolicyDelaySkipInd, self).__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(nn.Linear(env.observation_space.shape[0], 128), nn.LeakyReLU()))
        self.blocks.append(nn.Linear(env.observation_space.shape[0] + 128 * 1, env.action_space.n))

        # critic's layer
        self.critic = nn.Sequential(nn.Linear(env.observation_space.shape[0], 128), nn.LeakyReLU(), nn.Linear(128, 1))

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def get_init_hidden(self, obs):
        x = obs
        hidden_activations = []
        for block in self.blocks[:-1]:
            x = block(x)
            hidden_activations.append(x)
        x = torch.cat([obs] + hidden_activations, dim=0)
        x = self.blocks[-1](x)
        hidden_activations.append(x)
        return hidden_activations

    def forward(self, obs, hidden_acts=None):
        new_hidden = []
        for i, (input, block) in enumerate(zip([obs] + hidden_acts[:-1], self.blocks[:-1])):
            out = block(input)
            new_hidden.append(out)

        input = torch.cat([obs] + hidden_acts[:-1], dim=0)
        out = self.blocks[-1](input)
        new_hidden.append(out)
        action_prob = F.softmax(hidden_acts[-1], dim=-1)
        state_values = self.critic(obs)

        return action_prob, state_values, new_hidden