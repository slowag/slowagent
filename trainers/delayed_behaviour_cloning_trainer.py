import time
import torch
import numpy as np
import torch.nn.functional as F


def train_delayed_behaviour_cloning(args, envs, actor_target, actor, actor_optimizer, rb, writer, device):
    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    actor_updates = 0
    observations = torch.Tensor(obs).to(device)
    hidden_activations = actor.get_activations(observations)

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        actor.eval()
        with torch.no_grad():
            # target_actions, _, _ = actor_target.get_action(observations)
            target_actions, _, _ = actor_target.get_action(observations)
            actions, _, _, hidden_activations = actor.get_action(observations, hidden_activations)
        actor.train()
        actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add((obs, real_next_obs, target_actions.detach().cpu().numpy(), rewards, terminations))

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        observations = torch.Tensor(next_obs).to(device)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample_seq(args.N_hidden_layers + 2 + 1 + args.warm_up_seq, args.batch_size) # (args.N_hidden_layers + 2) is actual number of layer in NN, +1 is for the delay, +args.warm_up_seq is for the warm up sequence

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_iteration
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    actor_updates += 1
                    actor_optimizer.zero_grad()
                    action, _, _ = actor.learn_action(data.observations)
                    actor_loss = F.mse_loss(action, data.actions[-1])
                    actor_loss.backward()
                    actor.backward()
                    actor_optimizer.step()

            if global_step % 1000 == 0:
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("actor_updates", actor_updates, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"{args.out_dir}/model"
        print('Saving model to', model_path)
        torch.save(actor.state_dict(), model_path)

    if args.eval_model:
        print('evaluating model after training')
        from utils import evaluate_sac_delayed
        episodic_returns = evaluate_sac_delayed(actor, args, args.device)
        episodic_returns = np.array(episodic_returns)
        mean, std = np.mean(episodic_returns), np.std(episodic_returns)
        writer.add_scalar("eval/mean_episodic_return", mean)
        writer.add_scalar("eval/std_episodic_return", std)

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        print('evaluating model after training using mean')
        episodic_returns = evaluate_sac_delayed(actor, args, args.device, greedy=True)

        episodic_returns = np.array(episodic_returns)
        mean, std = np.mean(episodic_returns), np.std(episodic_returns)
        writer.add_scalar("eval/mean_mode_episodic_return", mean)
        writer.add_scalar("eval/std_mode_episodic_return", std)
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/mode_episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
