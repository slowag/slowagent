import torch
import numpy as np
import time
import torch.nn.functional as F
import cv2
import os
import seaborn as sns


def check_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")


def make_video_from_arrays(imgs, output_file, fps, size=None):
    if not imgs:
        raise ValueError("The list of images is empty.")
    if size is None:
        height, width, layers = imgs[0].shape
        size = (width, height)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, size)
    
    for img in imgs:
        if img.shape[1] != size[0] or img.shape[0] != size[1]:
            img = cv2.resize(img, size)
        video.write(img)
    
    video.release()
    print(f"Video saved as '{output_file}'")



def train_default(envs, actor, qf1_target, qf2_target, qf1, qf2, rb, log_alpha, alpha, q_optimizer, a_optimizer, actor_optimizer, target_entropy, device, writer, args):
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                print( f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}" )
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)
                # CRITIC training
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)

                # use Q-values only for the taken actions
                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
                qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                _, log_pi, action_probs = actor.get_action(data.observations)
                with torch.no_grad():
                    qf1_values = qf1(data.observations)
                    qf2_values = qf2(data.observations)
                    min_qf_values = torch.min(qf1_values, qf2_values)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()


def train_delayed_sac(envs, actor, qf1_target, qf2_target, qf1, qf2, rb, log_alpha, alpha, q_optimizer, a_optimizer, actor_optimizer, target_entropy, device, writer, args):

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    obs = torch.from_numpy(obs).to(device, dtype=torch.float32)
    hidden_activations = actor.get_zero_activations(obs)


    episode_states = []
    video_stamp= 10_000
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actor.eval()
            with torch.no_grad():
                actions, _, _, hidden_activations = actor.get_action(torch.Tensor(obs).to(device), hidden_activations)
            actor.train()
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        episode_states.append(next_obs)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                print( f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}" )
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                if 'MinAtar' in args.env_id:
                    if global_step > video_stamp and args.capture_video:
                        for i in range(len(episode_states)):
                            state = cv2.resize(np.array(episode_states[i][0], dtype=np.uint8), (128, 128), cv2.INTER_LINEAR)
                            n_channels = state.shape[-1]
                            cmap = sns.color_palette("cubehelix", n_channels)
                            cmap.insert(0, (0,0,0))
                            numerical_state = np.amax(
                                state * np.reshape(np.arange(n_channels) + 1, (1,1,-1)), 2)
                            rgb_array = np.stack(cmap)[numerical_state]
                            episode_states[i] = np.array(rgb_array*255, dtype=np.uint8)

                        name_vid = 'videos/{}/rl_video'.format(args.run_name) + str(global_step) + '.mp4'
                        check_folder_exists('videos/{}'.format(args.run_name))
                        make_video_from_arrays(episode_states, name_vid, 30)
                        video_stamp +=args.video_freq
                        writer.add_video('rl_video.mp4', torch.tensor(episode_states).permute(0, 3, 1, 2).unsqueeze(0), fps=30)
                        # writer.add_video('rl_video' + str(global_step) + '.mp4', torch.tensor(episode_states).permute(0, 3, 1, 2).unsqueeze(0), fps=30)
                episode_states.clear()
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add((obs, real_next_obs, actions, rewards, terminations))

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                data = rb.sample_seq(args.N_hidden_layers + 1 + 1, args.batch_size)
                # CRITIC training
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.learn_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations[-1])
                    qf2_next_target = qf2_target(data.next_observations[-1])
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = data.rewards[-1].flatten() + (1 - data.dones[-1].flatten()) * args.gamma * (min_qf_next_target)

                # use Q-values only for the taken actions
                qf1_values = qf1(data.observations[-1])
                qf2_values = qf2(data.observations[-1])

                qf1_a_values = qf1_values.gather(1, data.actions[-1].long())[:, 0]
                qf2_a_values = qf2_values.gather(1, data.actions[-1].long())[:, 0]
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                if global_step % args.policy_frequency == 0:

                    for _ in range(args.policy_iteration):
                        _, log_pi, action_probs = actor.learn_action(data.observations)
                        with torch.no_grad():
                            qf1_values = qf1(data.observations[-1])
                            qf2_values = qf2(data.observations[-1])
                            min_qf_values = torch.min(qf1_values, qf2_values)
                        # no need for reparameterization, the expectation can be calculated for discrete actions
                        actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        if args.autotune:
                            # re-use action probabilities for temperature loss
                            alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        if args.save_model and global_step > video_stamp:
            model_path = f"{args.out_dir}.actor.slowagent_model"
            qf1_path = f"{args.out_dir}.qf1.slowagent_model"
            qf2_path = f"{args.out_dir}.qf2.slowagent_model"
            print('Saving models to', model_path, qf1_path, qf2_path)
            torch.save(actor.state_dict(), model_path)
            torch.save(qf1.state_dict(), qf1_path)
            torch.save(qf2.state_dict(), qf2_path)

    if args.eval_model:
        print('evaluating model after training')
        from utils import evaluate_sac_discrete_delayed
        episodic_returns = evaluate_sac_discrete_delayed(actor, args, args.device)
        episodic_returns = np.array(episodic_returns)
        mean, std = np.mean(episodic_returns), np.std(episodic_returns)
        writer.add_scalar("eval/mean_episodic_return", mean)
        writer.add_scalar("eval/std_episodic_return", std)

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/mode_episodic_return", episodic_return, idx)

    envs.close()
    writer.close()