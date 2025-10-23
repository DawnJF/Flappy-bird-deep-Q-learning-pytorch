import os
import sys
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import logging
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from src.sac_policy.low_dimensional import Actor, SoftQNetwork
from src.buffers import ReplayBuffer
from src.utils import get_device, setup_logging

# from src.flappy_bird_env import FlappyBirdEnv


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    output_dir: str = "outputs/sac"
    seed: int = 1
    """seed of the experiment"""

    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""

    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    learning_starts: int = 5000
    """timestep to start learning"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""

    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    # Checkpoint arguments
    save_freq: int = 200000
    """frequency to save checkpoints"""
    load_checkpoint: str = None
    """path to load checkpoint from"""
    render: bool = False


device = get_device()


def make_env(env_id, capture_video, render=False):
    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{env_id}")
    else:
        env = gym.make(env_id, render_mode="human" if render else None)
    # env = SparseHopper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def save_checkpoint(actor, qf1, qf2, actor_optimizer, q_optimizer, global_step, args):
    """Save model checkpoint"""
    checkpoint_path = args.output_dir

    checkpoint = {
        "global_step": global_step,
        "actor_state_dict": actor.state_dict(),
        "qf1_state_dict": qf1.state_dict(),
        "qf2_state_dict": qf2.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "args": args,
    }

    torch.save(checkpoint, f"{checkpoint_path}/checkpoint_{global_step}.pt")
    print(f"Checkpoint saved at step {global_step}")


def load_checkpoint(checkpoint_path, actor, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    actor.load_state_dict(checkpoint["actor_state_dict"])
    # qf1.load_state_dict(checkpoint["qf1_state_dict"])
    # qf2.load_state_dict(checkpoint["qf2_state_dict"])
    # actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    # q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])

    return checkpoint["global_step"], checkpoint["args"]


def evaluate_agent(load_checkpoint_path, num_episodes=5, max_steps=4000):
    """Evaluate the agent and optionally render the environment"""

    env = make_env("Hopper-v4", False, render=True)
    print(f"action_space: {env.action_space}")
    print(f"observation_space: {env.observation_space}")

    actor = Actor(env.observation_space, env.action_space)

    load_checkpoint(load_checkpoint_path, actor, device="cpu")

    episode_returns = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, _, _ = actor.get_action(obs_tensor)
                action = action.cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)

            distance = info["x_position"]
            print(f"Step: {episode_length}, Reward: {reward}, distance: {distance}")

            reward_scalar = np.asarray(reward).item()
            episode_return += reward_scalar
            episode_length += 1
            done = terminated or truncated

            if done:
                print("Episode finished after {} timesteps".format(episode_length))

            # if render:
            # eval_env.render()
            # time.sleep(0.01)  # Small delay for better visualization

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        print(
            f"Eval Episode {episode + 1}: Return = {float(episode_return):.2f}, Length = {episode_length}"
        )

    env.close()

    avg_return = np.mean(episode_returns)
    avg_length = np.mean(episode_lengths)

    print(
        f"Evaluation Results: Avg Return = {float(avg_return):.2f}, Avg Length = {float(avg_length):.2f}"
    )
    return avg_return, avg_length


def train():

    args = tyro.cli(Args)
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time_str}"
    args.output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    setup_logging(args.output_dir)

    writer = SummaryWriter(args.output_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # env setup
    env = make_env(args.env_id, args.capture_video, render=args.render)

    logging.info(f"observation_space: {env.observation_space}")
    logging.info(f"action_space: {env.action_space}")
    logging.info(f"=" * 50)

    actor = Actor(env.observation_space, env.action_space).to(device)
    qf1 = SoftQNetwork(env.observation_space, env.action_space).to(device)
    qf2 = SoftQNetwork(env.observation_space, env.action_space).to(device)
    qf1_target = SoftQNetwork(env.observation_space, env.action_space).to(device)
    qf2_target = SoftQNetwork(env.observation_space, env.action_space).to(device)

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(env.action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
        logging.info(f"Target entropy set to {target_entropy:.2f}")
    else:
        alpha = args.alpha

    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    log_buffer_full = False

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):

        if global_step == args.learning_starts:
            logging.info("Learning starts now!")

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            action = np.array(env.action_space.sample())
        else:
            action, _, _ = actor.get_action(torch.Tensor(obs).unsqueeze(0).to(device))
            action = action.detach().cpu().numpy().squeeze()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, truncation, info = env.step(action)

        # TRY NOT TO MODIFY: record reward for plotting purposes
        if "final_info" in info:
            for info in info["final_info"]:
                if info is not None:
                    # logging.info(
                    #     f"global_step={global_step}, episodic_return={info['episode']['r']}"
                    # )
                    writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], global_step
                    )
                    break

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        if truncation and "final_observation" in info:  # Note
            real_next_obs = info["final_observation"]
        rb.add(obs, real_next_obs, action, reward, termination, info)

        if rb.size == args.buffer_size and not log_buffer_full:
            logging.info("Replay buffer is full!")
            log_buffer_full = True

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:

                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                writer.add_scalar("losses/entropy", -log_pi.mean().item(), global_step)

                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

                logging.info(
                    f"global_step={global_step}, actor_loss={actor_loss.item():.3f}, qf1_loss={qf1_loss.item():.3f}, qf2_loss={qf2_loss.item():.3f}, alpha={alpha:.3f}"
                )
                logging.info(f"SPS: {int(global_step / (time.time() - start_time))}")

        # Save checkpoint
        if global_step > 0 and global_step % args.save_freq == 0:
            save_checkpoint(
                actor,
                qf1,
                qf2,
                actor_optimizer,
                q_optimizer,
                global_step,
                args,
            )

    # Final checkpoint save
    save_checkpoint(actor, qf1, qf2, actor_optimizer, q_optimizer, global_step, args)

    env.close()
    writer.close()


def test():

    load_checkpoint_path = (
        "outputs/sac/Hopper-v4__sac__1__1757170278/checkpoint_200000.pt"
    )

    evaluate_agent(load_checkpoint_path, 1)


if __name__ == "__main__":
    test()
    # train()
