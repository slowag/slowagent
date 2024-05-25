import gymnasium as gym
from gymnasium.envs.mujoco import HalfCheetahEnv, AntEnv, Walker2dEnv, HopperEnv


class CustomHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self, render_mode='rgb_array', speed_up=1):
        super().__init__(render_mode=render_mode)
        self.model.opt.timestep = self.model.opt.timestep/speed_up

class CustomAnt(AntEnv):
    def __init__(self, speed_up):
        super().__init__()
        self.model.opt.timestep = self.model.opt.timestep/speed_up

class CustomWalker2d(Walker2dEnv):
    def __init__(self, speed_up):
        super().__init__()
        self.model.opt.timestep = self.model.opt.timestep/speed_up

class CustomHopper(HopperEnv):
    def __init__(self, speed_up):
        super().__init__()
        self.model.opt.timestep = self.model.opt.timestep/speed_up

gym.envs.registration.register(
    id='CustomHalfCheetah-v0',
    entry_point=CustomHalfCheetahEnv,
)

gym.envs.registration.register(
    id='CustomAnt-v0',
    entry_point=CustomAnt,
)

gym.envs.registration.register(
    id='CustomWalker2d-v0',
    entry_point=CustomWalker2d,
)

gym.envs.registration.register(
    id='CustomHopper-v0',
    entry_point=CustomHopper,
)



class FastRewardWrapper(gym.Wrapper):

    #n is the speed_up
    def __init__(self, env, max_steps=2000, speed_up=2):
        super(FastRewardWrapper, self).__init__(env)
        self.max_steps = max_steps
        self.speed_up = speed_up
        self.current_step = 0
        self.total_reward = 0
        self.reward_count = 0

    def reset(self, **kwargs):
        self.current_step = 0
        self.total_reward = 0
        self.reward_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1

        if self.current_step % self.speed_up == 0: 
            self.total_reward += reward
            self.reward_count += 1

        done = terminated or truncated or self.current_step >= self.max_steps
        return obs, reward, done, truncated, info

    def compute_return(self):
        if self.reward_count == 0:
            return 0
        return self.total_reward / self.reward_count