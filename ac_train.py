import argparse
import gym
import numpy as np

import torch
import torch.optim as optim


from torch.utils.tensorboard import SummaryWriter
import time

from nets import ac_nets

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')

parser.add_argument('--n_episodes', type=int, default=1000000)
parser.add_argument('--track', type=bool, default=False)
parser.add_argument('--exp_name', type=str)
parser.add_argument('--policy', type=str, default='Policy')
parser.add_argument('--trainer', type=str, default='ac')
parser.add_argument('--env_id', default='LunarLander-v2')
parser.add_argument('--wandb-entity', default='asynchronous-agent')
parser.add_argument('--wandb_project_name', default='asynchronous-agent')


if __name__ == '__main__':
    args = parser.parse_args()

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    env = gym.make(args.env_id)
    env.reset()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Policy = getattr(ac_nets, args.policy)
    model = Policy(env)
    optimizer = optim.Adam(model.parameters(), lr=3e-2)
    eps = np.finfo(np.float32).eps.item()

    if args.trainer == 'ac':
        from trainers.trainers_ac import ac_trainer
        ac_trainer(args, model, env, optimizer, writer)
    elif args.trainer == 'ac_delay':
        from trainers.trainers_ac import ac_trainer_delay
        ac_trainer_delay(args, model, env, optimizer, writer)
    else:
        raise ValueError(f'Invalid trainer {args.trainer}')