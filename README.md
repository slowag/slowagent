# Installation

The code was tested with python3.10.

Download Mujoco from [here](https://github.com/google-deepmind/mujoco/releases/download/2.3.3/mujoco-2.3.3-linux-x86_64.tar.gz) 
and place it in /home/$USER/.mujoco, and then source it:

```
export MUJOCO_PLUGIN_PATH=/home/$USER/.mujoco/mujoco-2.3.3/bin/mujoco_plugin/
```

and 

```
export MUJOCO_PATH=/home/$USER/.mujoco/mujoco-2.3.3/
```

Install dependencies with:

```
pip install -r requirements.txt
```


# Tranning Mujoco

Vanilla SAC is taken from [cleanRL](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)

To train the vanilla SAC algorithm run:

```
python train_sac.py --env_id HalfCheetah-v4
```

To train the agent without any skip-connection within an in-parallel execution framework and neuron execution time of 1 run:

```
python train_sac.py --env_id HalfCheetah-v4 --trainer delayed_action_buffer_hiddens --agent ActorSlow --frame_skip 1 
```

To train with projections to observation agent:

```
python train_sac.py --env_id HalfCheetah-v4 --trainer delayed_action_buffer_hiddens --agent ActorSlowConcat --frame_skip 1 
```

To train it with different neuron execution times, please change the `frame_skip` parameter to 2,3 or 4. 
Furthermore, change the `agent` parameter to `ActorSlowSkip` or `ActorSlowResSkip` to train with projections to action agent or with a residual agent, respectively.

# Tranning MinAtar
Install MinAtar using the instructions [here](https://github.com/kenjyoung/MinAtar).

For training Vanilla SAC (taken from [cleanRL](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py)), run: 
```
python train_sac_atari.py --env_id MinAtar/Breakout-v0 --trainer default --agent Actor
```


For training an agent without skip connections within an in-parallel execution and neuron execution time of 0.5 run:
```
python train_sac_atari.py --env_id MinAtar/Breakout-v0 --trainer delayed --agent ActorSlow --frame_skip 1 --repeat_frame 2
```


For training an agent with projections to action:

```
python train_sac_atari.py --env_id MinAtar/Breakout-v0 --trainer delayed --agent ActorSlowSkip --frame_skip 1 --repeat_frame 2
```

To train it with different neuron execution times, please change the `frame_skip` parameter to 2,3 or 4. 
Furthermore, the `agent` parameter is set to `ActorSlowSkipRes` to train with projections to action agent or with a residual agent, respectively.

# Speed-up estimation
To estimate the speed-up of the inference time on your GPU, please run the following command:

```
python test_in_parallel_speed_up.py
```