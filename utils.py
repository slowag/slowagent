import gymnasium as gym
from collections import deque
import numpy as np
from gymnasium.spaces import Box
import torch


def make_env_continuous(env_id, seed, idx, capture_video, run_name, args):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", speed_up=args.speed_up) if args.speed_up > 1 else gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, render_mode="rgb_array", speed_up=args.speed_up) if args.speed_up > 1 else gym.make(env_id, render_mode="rgb_array")
        if args.speed_up > 1:
            print(f"Speeding up the environment by a factor of {args.speed_up}")

        env = SkipFrameWrapper(env, frame_skip=args.frame_skip)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        if args.history_states > 1:
            env = FrameConcat(env, args.history_states)

        if args.num_last_actions > 0:
            env = LastActionWrapper(env, num_last_actions=args.num_last_actions)

        if args.normalize_observation:
            env = gym.wrappers.NormalizeObservation(env)

        env.action_space.seed(seed)
        return env

    return thunk


class LastActionWrapper(gym.Wrapper):
    def __init__(self, env, num_last_actions=1):
        super().__init__(env)

        self.num_last_actions = num_last_actions
        self.last_actions = deque(maxlen=num_last_actions)

        low, high = env.observation_space.low, env.observation_space.high
        obs_low = list(np.array(low).flatten())
        obs_high = list(np.array(high).flatten())

        action_low = list(np.array([env.action_space.low for _ in range(num_last_actions)]).squeeze().flatten())
        action_high = list(np.array([env.action_space.high for _ in range(num_last_actions)]).squeeze().flatten())

        low = np.array(obs_low + action_low)
        high = np.array(obs_high + action_high)
        self.observation_space = Box(low=low, high=high)

    def observation(self, obs):
        last_actions = np.concatenate(list(self.last_actions), axis=0)
        new_obs = np.concatenate([obs, last_actions])
        return new_obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        [self.last_actions.append(self.action_space.sample() * 0) for _ in range(self.num_last_actions)]
        return self.observation(obs), info

    def step(self, action):
        self.last_actions.append(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info


class FrameConcat(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        num_concat: int,
    ):
        gym.ObservationWrapper.__init__(self, env)

        self.num_concat = num_concat
        self.frames = deque(maxlen=num_concat)

        low, high = env.observation_space.low, env.observation_space.high
        low = np.array([[low] * num_concat]).squeeze().flatten()
        high = np.array([[high] * num_concat]).squeeze().flatten()

        self.observation_space = Box(low=low, high=high)

    def observation(self, ):
        return np.concatenate(list(self.frames), axis=0)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        [self.frames.append(obs) for _ in range(self.num_concat)]
        return self.observation(), info


class SkipFrameWrapper(gym.Wrapper):
    def __init__(self, env, frame_skip=1):
        super().__init__(env)
        self.frame_skip = frame_skip

    def step(self, action):
        total_reward = 0
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class SlowDownEnvWrapper(gym.Wrapper):
    def __init__(self, env, repeat_frame=1):
        super().__init__(env)
        self.repeat_frame = repeat_frame
        self.counter = 0

    def step(self, action):
        if self.counter % self.repeat_frame == 0:
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.last_res = obs, reward, terminated, truncated, info
        else:
            obs, reward, terminated, truncated, info = self.last_res
            reward = 0
            self.counter += 1

        return obs, reward, terminated, truncated, info


def atari_make_env(env_id, idx, capture_video, run_name, args):
    from stable_baselines3.common.atari_wrappers import (
        ClipRewardEnv,
        EpisodicLifeEnv,
        FireResetEnv,
        MaxAndSkipEnv,
        NoopResetEnv,
    )
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        if args.repeat_frame > 1:
            env = SlowDownEnvWrapper(env, repeat_frame=args.repeat_frame)

        env = NoopResetEnv(env, noop_max=30)
        if not args.remove_skip:
            env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(args.seed)
        return env

    return thunk


def mini_atari_make_env(env_id, idx, capture_video, run_name, args):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = SkipFrameWrapper(env, frame_skip=args.frame_skip)
        if args.repeat_frame > 1:
            env = SlowDownEnvWrapper(env, repeat_frame=args.repeat_frame)


        env.metadata['render_fps'] = 30
        return env

    return thunk


def minigrid_make_env(env_id, idx, capture_video, run_name, args):
    from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if args.repeat_frame > 1:
            env = SlowDownEnvWrapper(env, repeat_frame=args.repeat_frame)

        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

        return env

    return thunk


def make_env_discrete(env_id, idx, capture_video, run_name, args):

    if 'MinAtar' in env_id:
        args.inverse_channels = True
        return mini_atari_make_env(env_id, idx, capture_video, run_name, args)
    elif 'MiniGrid' in env_id:
        args.inverse_channels = False
        return minigrid_make_env(env_id, idx, capture_video, run_name, args)
    else:
        args.inverse_channels = False
        return atari_make_env(env_id, idx, capture_video, run_name, args)


def evaluate_sac(
    actor,
    args,
    device,
    eval_episodes: int = 100,
    greedy: bool = False,
):
    envs = gym.vector.SyncVectorEnv([make_env_continuous(args.env_id, args.seed, 0, args.capture_video_eval, args.run_name, args)])
    obs, _ = envs.reset(seed=args.seed)

    actor.eval()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            if not greedy:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            else:
                _, _, actions = actor.get_action(torch.Tensor(obs).to(device))
        obs, _, _, _, infos = envs.step(actions.cpu().numpy())

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
    actor.train()

    return episodic_returns


def evaluate_sac_delayed(
    actor,
    args,
    device,
    eval_episodes: int = 100,
    greedy: bool = False,
):
    envs = gym.vector.SyncVectorEnv([make_env_continuous(args.env_id, args.seed, 0, args.capture_video_eval, args.run_name, args)])
    obs, _ = envs.reset(seed=args.seed)
    hidden_activations = actor.get_activations(torch.Tensor(obs).to(device))

    actor.eval()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            if not greedy:
                actions, _, _, hidden_activations = actor.get_action(torch.Tensor(obs).to(device), hidden_activations)
            else:
                _, _, actions, hidden_activations = actor.get_action(torch.Tensor(obs).to(device), hidden_activations)
        obs, _, _, _, infos = envs.step(actions.cpu().numpy())

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
    actor.train()

    return episodic_returns


def evaluate_sac_discrete_delayed(
    actor,
    args,
    device,
    eval_episodes: int = 100,
    greedy: bool = False,
):
    envs = gym.vector.SyncVectorEnv([make_env_discrete(args.env_id, 0, True, args.run_name, args)])
    obs, _ = envs.reset(seed=args.seed)
    hidden_activations = actor.get_zero_activations(torch.Tensor(obs).to(device))

    actor.eval()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            if not greedy:
                actions, _, _, hidden_activations = actor.get_action(torch.Tensor(obs).to(device), hidden_activations)
            else:
                _, log_prob, _, hidden_activations = actor.get_action(torch.Tensor(obs).to(device), hidden_activations)
                actions = torch.argmax(log_prob, dim=-1)
        obs, _, _, _, infos = envs.step(actions.cpu().numpy())

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
    actor.train()

    return episodic_returns


def evaluate_ppo(
    actor,
    args,
    device,
    eval_episodes: int = 100,
    greedy: bool = False,
):
    envs = gym.vector.SyncVectorEnv([make_env_discrete(args.env_id, 0, True, args.run_name, args)])
    obs, _ = envs.reset(seed=args.seed)

    actor.eval()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            if not greedy:
                actions, _, _, _ = actor.get_action_and_value(torch.Tensor(obs).to(device))
            else:
                _, log_prob, _, _ = actor.get_action_and_value(torch.Tensor(obs).to(device))
                actions = torch.argmax(log_prob, dim=-1, keepdim=True)
        obs, _, _, _, infos = envs.step(actions.cpu().numpy())

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
    actor.train()

    return episodic_returns


def evaluate_ppo_delayed(
    actor,
    args,
    device,
    eval_episodes: int = 100,
    greedy: bool = False,
):
    envs = gym.vector.SyncVectorEnv([make_env_discrete(args.env_id, 0, True, args.run_name, args)])
    obs, _ = envs.reset(seed=args.seed)
    hidden_activations = actor.get_zero_activations(torch.Tensor(obs).to(device))

    actor.eval()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            if not greedy:
                actions, _, _, _, hidden_activations = actor.get_action_and_value(torch.Tensor(obs).to(device), hidden_activations)
            else:
                _, log_prob, _, _, hidden_activations = actor.get_action_and_value(torch.Tensor(obs).to(device), hidden_activations)
                actions = torch.argmax(log_prob, dim=-1, keepdim=True)
        obs, _, _, _, infos = envs.step(actions.cpu().numpy())

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
    actor.train()

    return episodic_returns
