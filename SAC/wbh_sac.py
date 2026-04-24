from common.normalization import Normalization, RewardScaling
from SAC.replay_buffer import ReplayBuffer
from SAC.sac import SACAgent
from common.env import (
    HoleBoardEnv,
    EnvConfig,
    SCALAR_DIM,
    PRIVILEGED_DIM,
    ACTION_DIM,
    IMG_H,
    IMG_W,
)
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
from collections import deque
import torch
import numpy as np
import time
import argparse
from isaacsim import SimulationApp
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SimulationApp must be started before all Isaac related imports
sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

matplotlib.use("Agg")


def evaluate_policy(env, agent, n_episodes=3):
    """Evaluation strategy: Turn off normalized updates and use deterministic actions."""
    env.training = False
    total_reward = 0.0
    term_keys = ["progress", "target_guide", "align", "post_align",
                 "pass", "collision", "smooth", "arrive", "alive"]
    term_sums = {k: 0.0 for k in term_keys}

    for _ in range(n_episodes):
        scalar_obs, _priv_obs, wrist_d, global_d, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(
                scalar_obs, wrist_d, global_d, deterministic=True)
            scalar_obs, _priv_obs, wrist_d, global_d, r, done, _info = env.step(
                action)
            ep_reward += r
            bd = env.reward_breakdown
            for k in term_keys:
                term_sums[k] += bd.get(k, 0.0)
        total_reward += ep_reward

    env.training = True
    avg_r = total_reward / n_episodes
    avg_terms = {k: term_sums[k] / n_episodes for k in term_keys}
    env._reward_manager.last_breakdown = avg_terms.copy()
    breakdown_str = "  ".join(f"{k}={v:+.4f}" for k, v in avg_terms.items())
    print(f"  [breakdown/eval_avg] {breakdown_str}")
    return avg_r


def save_training_plots(
    save_dir,
    actor_loss_log,
    critic_loss_log,
    alpha_log,
    eval_steps_log,
    eval_reward_log,
    stage_switch_steps,
):
    """Save training curve: actor/critic/alpha + eval reward。"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("SAC Training Curves", fontsize=14)

    ax = axes[0]
    if actor_loss_log:
        steps_a, vals_a = zip(*actor_loss_log)
        ax.plot(steps_a, vals_a, label="actor_loss",
                color="steelblue", linewidth=1)
    if critic_loss_log:
        steps_c, vals_c = zip(*critic_loss_log)
        ax.plot(steps_c, vals_c, label="critic_loss_mean",
                color="tomato", linewidth=1, linestyle="--")

    if alpha_log:
        steps_al, vals_al = zip(*alpha_log)
        ax2 = ax.twinx()
        ax2.plot(steps_al, vals_al, label="alpha",
                 color="darkgreen", linewidth=1, linestyle=":")
        ax2.set_ylabel("alpha", color="darkgreen")
        ax2.tick_params(axis="y", labelcolor="darkgreen")

    for s, st in stage_switch_steps:
        ax.axvline(x=s, color="gray", linestyle=":", linewidth=0.8)
        ax.text(s, ax.get_ylim()[1], f"s{st}",
                fontsize=7, color="gray", va="top")

    ax.set_xlabel("steps")
    ax.set_ylabel("loss")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if eval_reward_log:
        ax.plot(eval_steps_log, eval_reward_log, marker="o", markersize=3,
                color="seagreen", linewidth=1.2, label="eval_avg_reward")
        best_idx = int(np.argmax(eval_reward_log))
        ax.scatter(eval_steps_log[best_idx], eval_reward_log[best_idx],
                   color="gold", zorder=5, s=60, label=f"best={eval_reward_log[best_idx]:.2f}")

    for s, st in stage_switch_steps:
        ax.axvline(x=s, color="gray", linestyle=":", linewidth=0.8)
        ax.text(s, ax.get_ylim()[1], f"s{st}",
                fontsize=7, color="gray", va="top")

    ax.set_xlabel("steps")
    ax.set_ylabel("avg_reward")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] Training curve saved {out_path}")


def main(args):
    cfg = EnvConfig(
        headless=True,
        max_steps=args.max_episode_steps,
        debug_diagnostics=args.debug_diagnostics,
        debug_log_episodes=args.debug_log_episodes,
        debug_log_steps=args.debug_log_steps,
    )
    env = HoleBoardEnv(cfg, sim_app=sim_app)
    env.set_stage(args.stage)

    args.state_dim = SCALAR_DIM
    args.priv_dim = PRIVILEGED_DIM
    args.action_dim = ACTION_DIM
    args.img_h = IMG_H
    args.img_w = IMG_W
    args.max_action = 1.0

    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.save_dir}/stage{args.stage}/{run_id}"

    replay_buffer = ReplayBuffer(args)
    agent = SACAgent(args)
    writer = SummaryWriter(log_dir=f"runs/stage{args.stage}/{run_id}")

    if args.resume:
        prev_stage = agent.load(args.resume, reset_optimizer=True)
        print(
            f"[resume] Loaded model from {args.resume} (previous stage={prev_stage}), optimizer state cleared")
        if prev_stage != args.stage:
            agent.reset_lr()
            print(f"[stage switch] {prev_stage} → {args.stage}, lr reset")

    reward_norm = Normalization(shape=1) if args.use_reward_norm else None
    reward_scaling = RewardScaling(
        gamma=args.gamma) if args.use_reward_scaling else None

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    total_steps = 0
    episode_idx = 0
    evaluate_num = 0
    last_eval_steps = 0
    evaluate_rewards = []
    recent_rewards = deque(maxlen=5)
    alpha_freeze_episodes = 0
    stage = env.cfg.stage

    if args.stage > 1:
        alpha_freeze_episodes = 20
        agent.log_alpha.requires_grad_(False)
        print(
            f"[alpha] Initial stage={args.stage}, freezing entropy adjustment for {alpha_freeze_episodes} episodes")

    actor_loss_log = []
    critic_loss_log = []
    alpha_log = []
    eval_steps_log = []
    eval_reward_log = []
    stage_switch_steps = []

    print(
        f"Training started | stage={stage} | max_steps={args.max_train_steps}")

    while total_steps < args.max_train_steps:
        episode_idx += 1
        scalar_obs, priv_obs, wrist_d, global_d, _ = env.reset()

        new_stage = env.cfg.stage
        if new_stage != stage:
            print(
                f"[curriculum] stage {stage} → {new_stage}, reset optimizer momentum")
            agent.reset_lr()
            stage = new_stage
            stage_switch_steps.append((total_steps, stage))
            if reward_scaling is not None:
                reward_scaling.running_ms.n = 0
                reward_scaling.running_ms.mean = np.zeros(1)
                reward_scaling.running_ms.S = np.zeros(1)
                reward_scaling.running_ms.std = np.ones(1)
                reward_scaling.R = np.zeros(1)
                print(f"[reward_scaling] Stage switch, normalized statistics reset")
            replay_buffer.ptr = 0
            replay_buffer.size = 0
            print(
                f"[buffer] Stage switch, replay buffer cleared (to avoid contamination from previous stage Q targets)")
            alpha_freeze_episodes = 10
            agent.log_alpha.requires_grad_(False)
            print(
                f"[alpha] Stage switch, freezing entropy adjustment for {alpha_freeze_episodes} episodes")
        episode_steps = 0
        ep_reward = 0.0
        done = False

        while not done and total_steps < args.max_train_steps:
            episode_steps += 1

            if total_steps < args.start_steps:
                action = np.random.uniform(-1.0, 1.0,
                                           ACTION_DIM).astype(np.float32)
            else:
                action = agent.select_action(
                    scalar_obs, wrist_d, global_d, deterministic=False)

            scalar_obs_, priv_obs_, wrist_d_, global_d_, r, done, info = env.step(
                action)
            ep_reward += r

            if reward_scaling is not None:
                r = reward_scaling(r)
            elif reward_norm is not None:
                r = float(reward_norm(np.array([r]))[0])

            dw = done and not info.get("timeout", False)

            replay_buffer.store(
                scalar_obs,
                priv_obs,
                wrist_d,
                global_d,
                action,
                r,
                scalar_obs_,
                priv_obs_,
                wrist_d_,
                global_d_,
                dw,
            )

            scalar_obs = scalar_obs_
            priv_obs = priv_obs_
            wrist_d = wrist_d_
            global_d = global_d_
            total_steps += 1

            if replay_buffer.size >= args.batch_size and total_steps >= args.update_after:
                for _ in range(args.updates_per_step):
                    metrics = agent.update(replay_buffer)
                if metrics is not None:
                    critic_loss_mean = 0.5 * \
                        (metrics["critic1_loss"] + metrics["critic2_loss"])
                    writer.add_scalar("train/actor_loss",
                                      metrics["actor_loss"], total_steps)
                    writer.add_scalar("train/critic1_loss",
                                      metrics["critic1_loss"], total_steps)
                    writer.add_scalar("train/critic2_loss",
                                      metrics["critic2_loss"], total_steps)
                    writer.add_scalar("train/alpha_loss",
                                      metrics["alpha_loss"],      total_steps)
                    writer.add_scalar("train/alpha",
                                      metrics["alpha"],           total_steps)
                    writer.add_scalar("train/aux_loss",
                                      metrics["aux_loss"],        total_steps)
                    writer.add_scalar("train/wrist_feat_std",
                                      metrics["wrist_feat_std"],  total_steps)
                    writer.add_scalar("train/global_feat_std",
                                      metrics["global_feat_std"], total_steps)
                    writer.add_scalar("train/wrist_gnorm",
                                      metrics["wrist_gnorm"],     total_steps)
                    writer.add_scalar("train/global_gnorm",
                                      metrics["global_gnorm"],    total_steps)
                    actor_loss_log.append((total_steps, metrics["actor_loss"]))
                    critic_loss_log.append((total_steps, critic_loss_mean))
                    alpha_log.append((total_steps, metrics["alpha"]))

            if total_steps % args.save_interval == 0 and total_steps > 0:
                ckpt_path = f"{save_dir}/ckpt_{total_steps}.pt"
                agent.save(ckpt_path, stage=stage,
                           obs_norm=env.scalar_norm, priv_norm=env.priv_norm)
                print(f"[checkpoint] Saved {ckpt_path}")

        recent_rewards.append(ep_reward)
        reward_avg5 = float(np.mean(recent_rewards))
        writer.add_scalar("train/reward_avg5", reward_avg5, total_steps)
        writer.add_scalar("train/stage", stage, total_steps)
        print(
            f"[stage {stage}][episode {episode_idx}] done | steps={episode_steps} | ep_r={ep_reward:.3f} | avg5={reward_avg5:.3f}")
        if reward_scaling is not None:
            reward_scaling.reset()

        if alpha_freeze_episodes > 0:
            alpha_freeze_episodes -= 1
            if alpha_freeze_episodes == 0:
                agent.log_alpha.requires_grad_(True)
                print("[alpha] Freeze period ended, resuming entropy adjustment")

        if total_steps - last_eval_steps >= args.evaluate_freq:
            last_eval_steps = total_steps
            evaluate_num += 1
            avg_r = evaluate_policy(env, agent, n_episodes=3)
            evaluate_rewards.append(avg_r)
            eval_steps_log.append(total_steps)
            eval_reward_log.append(avg_r)
            writer.add_scalar("eval/episode_reward", avg_r, total_steps)
            print(
                f"[eval {evaluate_num}] steps={total_steps}  avg_reward={avg_r:.3f}")

            save_training_plots(
                save_dir,
                actor_loss_log,
                critic_loss_log,
                alpha_log,
                eval_steps_log,
                eval_reward_log,
                stage_switch_steps,
            )

            if avg_r >= max(evaluate_rewards):
                best_path = f"{save_dir}/best.pt"
                agent.save(best_path, stage=stage,
                           obs_norm=env.scalar_norm, priv_norm=env.priv_norm)
                print(f"[best] avg_r={avg_r:.3f}, saved {best_path}")

    final_path = f"{save_dir}/final.pt"
    agent.save(final_path, stage=stage,
               obs_norm=env.scalar_norm, priv_norm=env.priv_norm)
    print(f"[final] Saved {final_path}")

    save_training_plots(
        save_dir,
        actor_loss_log,
        critic_loss_log,
        alpha_log,
        eval_steps_log,
        eval_reward_log,
        stage_switch_steps,
    )

    env.close()
    writer.close()
    print("Training Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ME5406 SAC Training")

    # Save / Resume
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_interval", type=int,
                        default=2000, help="Interval (steps) to save a checkpoint")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to resume training from")

    # Training Scale
    parser.add_argument("--max_train_steps", type=int, default=int(2e6))
    parser.add_argument("--max_episode_steps", type=int, default=400)
    parser.add_argument("--evaluate_freq", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", type=int, default=1,
                        help="Initial curriculum stage 1/2/3")

    # SAC Hyperparameters
    parser.add_argument("--aux_coef", type=float,
                        default=0.3, help="Weight for CNN auxiliary loss")
    parser.add_argument("--replay_size", type=int, default=int(1e5))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-4)
    parser.add_argument("--lr_alpha", type=float, default=3e-4)
    parser.add_argument("--init_alpha", type=float, default=0.2)
    parser.add_argument("--start_steps", type=int,
                        default=5000, help="Number of initial steps for random exploration")
    parser.add_argument("--update_after", type=int,
                        default=1000, help="Step count to start network updates")
    parser.add_argument("--updates_per_step", type=int,
                        default=1, help="Number of gradient updates per environment interaction step")

    # Reward Processing (Choose one)
    parser.add_argument("--use_reward_norm", type=bool, default=False)
    parser.add_argument("--use_reward_scaling", type=bool, default=True)

    # Debug
    parser.add_argument("--debug_diagnostics", type=bool, default=False)
    parser.add_argument("--debug_log_episodes", type=int, default=3)
    parser.add_argument("--debug_log_steps", type=int, default=5)

    args = parser.parse_args()
    main(args)
