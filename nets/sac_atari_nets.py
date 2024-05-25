import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions.categorical import Categorical
import numpy as np


def transform_obs(x, env_id):
    if 'MinAtar' in env_id:
        x = x.permute(0, 3, 1, 2).float()
    elif 'MiniGrid' in env_id:
        x = x.permute(0, 3, 1, 2).float() / 255.0
    else:
        x = x / 255.0
    return x

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_orthogonal(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.args = args
        print(' self.args.env_id',  self.args.env_id)
        if 'MinAtar' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, kernel_size=3, padding='same')),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=3, padding='same')),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, padding='same')),
                nn.Flatten()
            )
            with torch.inference_mode():
                output_dim = self.conv(torch.zeros(1, obs_shape[-1], obs_shape[0], obs_shape[1])).shape[1]
            self.fc1 = layer_init(nn.Linear(output_dim, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))
        elif 'MiniGrid' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                nn.Sequential(layer_init(nn.Conv2d(obs_shape[-1], 32, 8, stride=4)),
                              nn.ReLU()),
                nn.Sequential(layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                              nn.ReLU()),
                nn.Sequential(layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                              nn.ReLU()),
                nn.Flatten(),
            )
            with torch.inference_mode():
                output_dim = self.conv(torch.zeros(1, obs_shape[-1], obs_shape[0], obs_shape[1])).shape[1]
            self.fc1 = layer_init(nn.Linear(output_dim, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

        else:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                nn.Flatten(),
            )
            with torch.inference_mode():
                output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

            self.fc1 = layer_init(nn.Linear(output_dim, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        x = transform_obs(x, self.args.env_id)
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


class SoftQNetwork(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.args = args
        if 'MinAtar' in args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, kernel_size=3, padding='same')),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=3, padding='same')),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, padding='same')),
                nn.Flatten(),
            )
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            with torch.inference_mode():
                output_dim = self.conv(torch.zeros(1, obs_shape[-1], 10, 10)).shape[1]

            self.fc1 = layer_init(nn.Linear(output_dim, 512))
            self.fc_q = layer_init(nn.Linear(512, envs.single_action_space.n))
        elif 'MiniGrid' in args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, kernel_size=8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                nn.Flatten(),
            )
            with torch.inference_mode():
                output_dim = self.conv(torch.zeros(1, obs_shape[-1], obs_shape[0], obs_shape[1])).shape[1]

            self.fc1 = layer_init(nn.Linear(output_dim, 512))
            self.fc_q = layer_init(nn.Linear(512, envs.single_action_space.n))
        else:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                nn.Flatten(),
            )

            with torch.inference_mode():
                output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

            self.fc1 = layer_init(nn.Linear(output_dim, 512))
            self.fc_q = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        x = transform_obs(x, self.args.env_id)
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals


class ActorSlow(Actor):
    def __init__(self, envs, args):
        nn.Module.__init__(self)
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if 'MinAtar' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(32, 64, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, padding='same')),
            )
            
            with torch.no_grad():
                dem = torch.zeros((1,obs_shape[-1], obs_shape[0], obs_shape[1]))
                shape = np.array(dem.shape).prod()
                for block in list(self.conv[:-1]):
                    dem = block(dem)
                    shape=np.array(dem.shape).prod()
            self.fc1 = layer_init(nn.Linear(shape, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

        elif 'MiniGrid' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, 8, stride=4)),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            )
            with torch.no_grad():
                dem = torch.zeros((1,obs_shape[-1], obs_shape[0], obs_shape[1]))
                shape = np.array(dem.shape).prod()
                for block in list(self.conv):
                    dem = block(dem)
                    shape=np.array(dem.shape).prod()
            self.fc1 = layer_init(nn.Linear(shape, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

        else:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            )
            
            with torch.no_grad():
                dem = torch.zeros((1,*obs_shape))
                shape = np.array(dem.shape).prod()
                for block in list(self.conv):
                    dem = block(dem)
                    shape=np.array(dem.shape).prod()
            self.fc1 = layer_init(nn.Linear(shape, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

    def init_activation(self, x):
        if self.args.get_instant_activations:
            return x
        else:
            return torch.zeros_like(x)

    def get_zero_activations(self, x):
        x = transform_obs(x, self.args.env_id)

        hidden_activations = []
        for block in list(self.conv[:-1]):
            x = F.relu(block(x))
            hidden_activations.append(self.init_activation(x))
        block = list(self.conv)[-1]
        x = nn.Flatten()(F.relu(block(x)))
        hidden_activations.append(self.init_activation(x))
        return hidden_activations
    
    def forward(self, x, hidden_acts=None):
        new_hidden = []
        for input, block in zip([x] + hidden_acts[:-2], list(self.conv)[:-1]):
            out = F.relu(block(input))
            new_hidden.append(out)
        
        last_hidden_n = list(self.conv)[-1](hidden_acts[-2])
        last_hidden_n = nn.Flatten()(last_hidden_n)
        new_hidden.append(last_hidden_n)

        logits  = self.fc_logits(F.relu(self.fc1(hidden_acts[-1])))
        return logits, new_hidden

    def get_action(self, x, hidden_acts=None):
        x = transform_obs(x, self.args.env_id)
        logits, new_hidden = self(x, hidden_acts)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs, new_hidden

    def learn_action(self, obs):
        hidden_activations = self.get_zero_activations(obs[0])
        for i, ob in enumerate(obs):
            action, log_prob, mean, hidden_activations = self.get_action(ob, hidden_activations)
        return action, log_prob, mean


class ActorSlowConcat(ActorSlow):
    def __init__(self, envs, args):
        nn.Module.__init__(self)
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if 'MinAtar' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(32 + obs_shape[-1], 64, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(64 + obs_shape[-1], 64, kernel_size=3, padding='same')),
            )

            self.fc1 = layer_init(nn.Linear(6400, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

        elif 'MiniGrid' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, 8, stride=4)),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            )
            self.fc1 = layer_init(nn.Linear(100, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

        else:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            )

            self.fc1 = layer_init(nn.Linear(100, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

    def get_zero_activations(self, x):
        x = transform_obs(x, self.args.env_id)
        obs = x

        hidden_activations = []
        for block in list(self.conv[:-1]):
            x = F.relu(block(x))
            hidden_activations.append(self.init_activation(x))
            obs_int = F.interpolate(obs, (x.shape[2], x.shape[3]))
            x = torch.cat([x, obs_int], dim=1)
        block = list(self.conv)[-1]
        x = nn.Flatten()(F.relu(block(x)))
        hidden_activations.append(self.init_activation(x))
        return hidden_activations

    def forward(self, x, hidden_acts=None):
        new_hidden = []
        out = F.relu(self.conv[0](x))
        new_hidden.append(out)

        for input, block in zip(hidden_acts[:-2], list(self.conv)[1:-1]):
            obs_int = F.interpolate(x, (input.shape[2], input.shape[3]))
            input = torch.cat([obs_int, input], dim=1)
            out = F.relu(block(input))
            new_hidden.append(out)

        obs_int = F.interpolate(x, (hidden_acts[-2].shape[2], hidden_acts[-2].shape[3]))
        input = torch.cat([hidden_acts[-2], obs_int], dim=1)
        last_hidden_n = list(self.conv)[-1](input)
        last_hidden_n = nn.Flatten()(last_hidden_n)
        new_hidden.append(last_hidden_n)

        logits = self.fc_logits(F.relu(self.fc1(hidden_acts[-1])))
        return logits, new_hidden


class ActorSlowConcatLSTM(ActorSlow):
    def __init__(self, envs, args):
        nn.Module.__init__(self)
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if 'MinAtar' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(32 + obs_shape[-1], 64, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(64 + obs_shape[-1], 64, kernel_size=3, padding='same')),
            )

            self.fc1 = layer_init(nn.Linear(6400, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

        elif 'MiniGrid' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, 8, stride=4)),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            )
            self.fc1 = layer_init(nn.Linear(100, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

        else:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            )

            self.fc1 = layer_init(nn.Linear(100, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

    def get_zero_activations(self, x):
        x = transform_obs(x, self.args.env_id)
        obs = x

        hidden_activations = []
        for block in list(self.conv[:-1]):
            x = F.relu(block(x))
            hidden_activations.append(self.init_activation(x))
            obs_int = F.interpolate(obs, (x.shape[2], x.shape[3]))
            x = torch.cat([x, obs_int], dim=1)
        block = list(self.conv)[-1]
        x = nn.Flatten()(F.relu(block(x)))
        hidden_activations.append(self.init_activation(x))
        return hidden_activations

    def forward(self, x, hidden_acts=None):
        new_hidden = []
        out = F.relu(self.conv[0](x))
        new_hidden.append(out)

        for input, block in zip(hidden_acts[:-2], list(self.conv)[1:-1]):
            obs_int = F.interpolate(x, (input.shape[2], input.shape[3]))
            input = torch.cat([obs_int, input], dim=1)
            out = F.relu(block(input))
            new_hidden.append(out)

        obs_int = F.interpolate(x, (hidden_acts[-2].shape[2], hidden_acts[-2].shape[3]))
        input = torch.cat([hidden_acts[-2], obs_int], dim=1)
        last_hidden_n = list(self.conv)[-1](input)
        last_hidden_n = nn.Flatten()(last_hidden_n)
        new_hidden.append(last_hidden_n)

        logits = self.fc_logits(F.relu(self.fc1(hidden_acts[-1])))
        return logits, new_hidden


class ActorSlowSkip(ActorSlow):
    def __init__(self, envs, args):
        nn.Module.__init__(self)
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if 'MinAtar' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(32, 64, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, padding='same')),
            )
            
            with torch.no_grad():
                dem = torch.zeros((1,obs_shape[-1], obs_shape[0], obs_shape[1]))
                shape = np.array(dem.shape).prod()
                for block in list(self.conv[:-1]):
                    dem = block(dem)
                    shape+=np.array(dem.shape).prod()

            self.fc1 = layer_init(nn.Linear(shape, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

        elif 'MiniGrid' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, 8, stride=4)),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            )

            with torch.no_grad():
                dem = torch.zeros((1,obs_shape[-1], obs_shape[0], obs_shape[1]))
                outs = []
                for block in list(self.conv[:-1]):
                    dem = block(dem)
                    outs.append(dem)

                spatial = outs[-1].shape[-1]
                shape = np.array([1, obs_shape[-1], spatial, spatial]).prod()
                for out in outs:
                    max_pool_size = out.shape[-1] // spatial
                    max_pooled = nn.MaxPool2d(max_pool_size)(out)
                    out = F.interpolate(max_pooled, spatial)
                    shape += np.array(out.shape).prod()

            self.fc1 = layer_init(nn.Linear(shape, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

        else:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            )
            
            with torch.no_grad():
                dem = torch.zeros((1,*obs_shape))
                outs = []
                for block in list(self.conv[:-1]):
                    dem = block(dem)
                    outs.append(dem)

                spatial = outs[-1].shape[-1]
                shape = np.array([1,obs_shape[0], spatial, spatial]).prod()
                for out in outs:
                    max_pool_size = out.shape[-1]//spatial
                    max_pooled = nn.MaxPool2d(max_pool_size)(out)
                    out = F.interpolate(max_pooled, spatial)
                    shape+=np.array(out.shape).prod()

            self.fc1 = layer_init(nn.Linear(shape, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

    def get_zero_activations(self, x):
        x = transform_obs(x, self.args.env_id)
        orig_x = x
        hidden_activations = []
        for block in list(self.conv[:-1]):
            x = F.relu(block(x))
            hidden_activations.append(self.init_activation(x))

        reshaped = [] 
        all_items = [orig_x] + hidden_activations
        last_spatial_size = all_items[-1].shape[-1] 
        for item in all_items:
            max_pool_size = item.shape[-1]//last_spatial_size
            max_pooled = nn.MaxPool2d(max_pool_size)(item)
            item = F.interpolate(max_pooled, last_spatial_size)
            reshaped.append(nn.Flatten()(item))
            if self.args.skip_detach:
                item.detach()

        x = torch.cat(reshaped, dim=1)
        hidden_activations.append(self.init_activation(x))

        return hidden_activations
    
    def forward(self, x, hidden_acts=None):
        new_hidden = []
        for input, block in zip([x] + hidden_acts[:-2], list(self.conv)):
            out = F.relu(block(input))
            new_hidden.append(out)
        
        reshaped = []
        last_spatial_size = hidden_acts[-2].shape[-1]
        for item in ([x] + hidden_acts[:-1]):
            max_pool_size = item.shape[-1]//last_spatial_size
            max_pooled = nn.MaxPool2d(max_pool_size)(item)
            item = F.interpolate(max_pooled, last_spatial_size)
            reshaped.append(nn.Flatten()(item))
            if self.args.skip_detach:
                item.detach()

        last_hidden_n = torch.cat(reshaped, dim=1)
        new_hidden.append(last_hidden_n)
        logits  = self.fc_logits(F.relu(self.fc1(hidden_acts[-1])))

        return logits, new_hidden


class ActorSlowSkipRes(ActorSlowSkip):
    def __init__(self, envs, args):
        nn.Module.__init__(self)
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if 'MinAtar' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(32+32, 64, kernel_size=3, padding='same')),
                layer_init(nn.Conv2d(64+64, 64, kernel_size=3, padding='same')),
            )
            
            with torch.no_grad():
                x = torch.zeros((1,obs_shape[-1], obs_shape[0], obs_shape[1]))
                shape = np.array(x.shape).prod()
                res = torch.zeros_like(x)
                for i,block in enumerate(list(self.conv)):
                    if i==0:
                        x = F.relu(block(x))
                        res = x
                    else:
                        x = F.relu(block(torch.cat([x, F.interpolate(res, x.shape[-1])],1)))
                        res = x
                    shape+=np.array(x.shape).prod()

            self.fc1 = layer_init(nn.Linear(shape, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

        elif 'MiniGrid' in self.args.env_id:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[-1], 32, 8, stride=4)),
                layer_init(nn.Conv2d(32+32, 64, 4, stride=2)),
                layer_init(nn.Conv2d(64+64, 64, 3, stride=1)),
            )

            with torch.no_grad():
                x = torch.zeros((1,obs_shape[-1], obs_shape[0], obs_shape[1]))
                outs = []
                res = torch.zeros_like(x)
                for i,block in enumerate(list(self.conv)):
                    if i==0:
                        x = F.relu(block(x))
                        res = x
                    else:
                        x = F.relu(block(torch.cat([x, F.interpolate(res, x.shape[-1])],1)))
                        res = x
                    outs.append(torch.zeros_like(x))

                spatial = outs[-1].shape[-1]
                shape = np.array([1,obs_shape[-1], spatial, spatial]).prod()
                for out in outs:
                    max_pool_size = out.shape[-1]//spatial
                    max_pooled = nn.MaxPool2d(max_pool_size)(out)
                    out = F.interpolate(max_pooled, spatial)
                    shape+=np.array(out.shape).prod()

            self.fc1 = layer_init(nn.Linear(shape, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

        else:
            obs_shape = envs.single_observation_space.shape
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                layer_init(nn.Conv2d(32 + 32 , 64, kernel_size=4, stride=2)),
                layer_init(nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1)),
            )
            
            with torch.no_grad():
                x = torch.zeros((1,*obs_shape))
                outs = []
                res = torch.zeros_like(x)
                for i,block in enumerate(list(self.conv)):
                    if i==0:
                        x = F.relu(block(x))
                        res = x
                    else:
                        x = F.relu(block(torch.cat([x, F.interpolate(res, x.shape[-1])],1)))
                        res = x
                    outs.append(torch.zeros_like(x))

                spatial = outs[-1].shape[-1]
                shape = np.array([1,obs_shape[0], spatial, spatial]).prod()
                for out in outs:
                    max_pool_size = out.shape[-1]//spatial
                    max_pooled = nn.MaxPool2d(max_pool_size)(out)
                    out = F.interpolate(max_pooled, spatial)
                    shape+=np.array(out.shape).prod()

            self.fc1 = layer_init(nn.Linear(shape, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

    def get_zero_activations(self, x):
        x = transform_obs(x, self.args.env_id)
        
        orig_x = x
        hidden_activations = []

        res = torch.zeros_like(x)
        for i,block in enumerate(list(self.conv)):
            if i==0:
                x = F.relu(block(x))
                res = x
            else:
                x = F.relu(block(torch.cat([x, F.interpolate(res, x.shape[-1])],1)))
                res = x
            hidden_activations.append(self.init_activation(x))

        reshaped = [] 
        all_items = [orig_x] + hidden_activations
        last_spatial_size = all_items[-1].shape[-1] 
        for item in all_items:
            max_pool_size = item.shape[-1]//last_spatial_size
            max_pooled = nn.MaxPool2d(max_pool_size)(item)
            item = F.interpolate(max_pooled, last_spatial_size)
            reshaped.append(nn.Flatten()(item))
            if self.args.skip_detach:
                item.detach()

        x = torch.cat(reshaped, dim=1)
        hidden_activations.append(self.init_activation(x))

        return hidden_activations
    
    def forward(self, x, hidden_acts=None):
        new_hidden = []
        res = torch.zeros_like(x)
        for i, (input, block) in enumerate(zip([x] + hidden_acts[:-2], list(self.conv))):
            if i==0:
                out = F.relu(block(x))
                res = out
            else:
                out = F.relu(block(torch.cat([input, F.interpolate(res, out.shape[-1])],1)))
                res = out
            new_hidden.append(out)

        reshaped = []
        last_spatial_size = hidden_acts[-2].shape[-1]
        for item in ([x] + hidden_acts[:-1]):
            max_pool_size = item.shape[-1]//last_spatial_size
            max_pooled = nn.MaxPool2d(max_pool_size)(item)
            item = F.interpolate(max_pooled, last_spatial_size)
            reshaped.append(nn.Flatten()(item))
            if self.args.skip_detach:
                item.detach()

        last_hidden_n = torch.cat(reshaped, dim=1)
        new_hidden.append(last_hidden_n)
        logits  = self.fc_logits(F.relu(self.fc1(hidden_acts[-1])))

        return logits, new_hidden
