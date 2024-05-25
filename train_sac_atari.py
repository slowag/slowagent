import os
import random
import time
from dataclasses import dataclass
import json

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from replay_buffer import ReplayMemoryDisc
from torch.utils.tensorboard import SummaryWriter
from nets.sac_atari_nets import SoftQNetwork
import utils

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "asynchronous-agent"
    """the wandb's project name"""
    wandb_entity: str = 'asynchronous-agent'
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MinAtar/Breakout-v0"
    """the id of the environment"""
    total_timesteps: int = 1_300_000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""  # smaller than in original paper but evaluation is done only for 100k steps anyway
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2e4
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates"""
    target_network_frequency: int = 8000
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89*0.5
    """coefficient for scaling the autotune entropy target"""
    agent: str = 'Actor'
    """the agent type"""
    repeat_frame: int = 2
    """repeat frame to slow down env"""
    N_hidden_layers: int = 2
    """number of hidden layers in NN"""
    policy_frequency: int = 8
    """policy update frequency"""
    policy_iteration: int = 1
    """policy update iteration"""
    trainer: str = 'delayed'
    """trainer type"""
    skip_detach: bool = False
    remove_skip: bool = False
    """remove skip frame wrapper in atairi env"""
    delay_degug: int = 5
    """delay for debugging agent"""
    save_model: bool = False
    """save the model"""
    eval_model: bool = True
    """evaluate the model"""
    frame_skip: int = 1
    """how many frames to skip while processing one layer of NN; speed of env = frame_skip * speed of neuron firing"""
    get_instant_activations: bool = False
    """get nonzero initial activation with an instance forward pass of the agent"""

    video_freq=50_000
    '''video freq for min atari'''


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    group_name = f"{args.agent}_fs{args.frame_skip}_rf{args.repeat_frame}"
    if args.exp_name != "":
        group_name = f"{group_name}_{args.exp_name}"
    run_name = f"{group_name}_s{args.seed}_{int(time.time())}"
    args.group_name = group_name
    args.run_name = run_name
    args.randint = random.randint(0, 1000)
    args.out_dir = f"runs/{args.env_id}/{args.group_name}/{run_name}_randint{args.randint}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            group=group_name,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(args.out_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    with open(f'{args.out_dir}/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.device = device

    # env setup
    envs = gym.vector.SyncVectorEnv([utils.make_env_discrete(args.env_id, 0, args.capture_video, run_name, args)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    qf1 = SoftQNetwork(envs, args).to(device)
    qf2 = SoftQNetwork(envs, args).to(device)
    qf1_target = SoftQNetwork(envs, args).to(device)
    qf2_target = SoftQNetwork(envs, args).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    try:
        import nets.sac_atari_nets as nets
        Actor = getattr(nets, args.agent)
        actor = Actor(envs, args).to(device)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)
    except AttributeError:
        raise ValueError(f"unknown agent type {args.agent}")
    print('actor', actor)

    if args.trainer=='default':
        from trainers.trainers_sac_atari import train_default
        from stable_baselines3.common.buffers import ReplayBuffer
        rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        )
        train_default(envs, actor, qf1_target, qf2_target, qf1, qf2, rb, log_alpha, alpha, q_optimizer, a_optimizer, actor_optimizer, target_entropy, device, writer, args)

    elif args.trainer=='delayed':
        from trainers.trainers_sac_atari import train_delayed_sac
        rb = ReplayMemoryDisc(
            args.buffer_size,
            envs.single_observation_space.shape,
            envs.single_action_space.n,
            envs.single_observation_space.dtype,
            device=device,
        )
        train_delayed_sac(envs, actor, qf1_target, qf2_target, qf1, qf2, rb, log_alpha, alpha, q_optimizer, a_optimizer, actor_optimizer, target_entropy, device, writer, args)
    else:
        raise ValueError(f"unknown trainer type {args.trainer}")

        