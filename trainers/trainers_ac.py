import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from collections import namedtuple


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


def finish_episode(args, model, optimizer, eps=1e-12):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def ac_trainer(args, model, env, optimizer, writer):
    def select_action(state):
        state = torch.from_numpy(state).float()
        probs, state_value = model(state)
        m = Categorical(probs)
        action = m.sample()
        model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

    running_reward = 0
    for i_episode in range(args.n_episodes):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode(args, model, optimizer)
        writer.add_scalar("rewards/mean_reward", running_reward, i_episode)
        writer.add_scalar("rewards/i_episode", i_episode, i_episode)

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))


def ac_trainer_delay(args, model, env, optimizer, writer):
    def select_action_delay(state, hidden_act):
        state = torch.from_numpy(state).float()
        probs, state_value, new_hidden_acts = model(state, hidden_act)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        # the action to take (left or right)
        return action.item(), new_hidden_acts

    running_reward = 10

    for i_episode in range(args.n_episodes):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        hidden_acts = model.get_init_hidden(torch.tensor(state, dtype=torch.float32))
        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):
            # select action from policy
            action, hidden_acts = select_action_delay(state, hidden_acts)

            # take the action
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        writer.add_scalar("rewards/mean_reward", running_reward, i_episode)
        writer.add_scalar("rewards/i_episode", i_episode, i_episode)

        # perform backprop
        finish_episode(args, model, optimizer)

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))