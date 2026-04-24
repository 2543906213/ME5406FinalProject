from TD3.replaybuffer_TD3 import ReplayBuffer
from TD3.TD3 import TD3
from common.normalization import Normalization, RewardScaling
from common.env import HoleBoardEnv, EnvConfig, SCALAR_DIM, PRIVILEGED_DIM, ACTION_DIM, IMG_H, IMG_W
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
import time
import argparse
import numpy as np
import torch
from isaacsim import SimulationApp
import sys
import os

# Ensure the 'common' folder in the root directory can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# !! SimulationApp MUST be started before all other imports !!
sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

matplotlib.use("Agg")

# === Core Imports: Pointing to the TD3 folder and public common folder ===

# -----------------------------------------------------------------------------
# Policy Evaluation (No noise added during TD3 evaluation)
# -----------------------------------------------------------------------------


def evaluate_policy(env, agent, n_episodes=10, stage=1):
    env.training = False
    total_reward = 0.0
    term_keys = ["progress", "target_guide", "align",
                 "pass", "collision", "smooth", "arrive", "alive"]
    term_sums = {k: 0.0 for k in term_keys}

    for _ in range(n_episodes):
        scalar_obs, priv_obs, wrist_d, global_d, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            # TD3 evaluation calls select_action directly, which contains no exploration noise
            action = agent.select_action(scalar_obs, wrist_d, global_d)
            scalar_obs, priv_obs, wrist_d, global_d, r, done, _ = env.step(
                action)
            ep_reward += r
            bd = env.reward_breakdown
            for k in term_keys:
                term_sums[k] += bd.get(k, 0.0)
        total_reward += ep_reward

    env.training = True
    avg_r = total_reward / n_episodes
    avg_terms = {k: term_sums[k] / n_episodes for k in term_keys}
    breakdown_str = "  ".join(f"{k}={v:+.4f}" for k, v in avg_terms.items())
    print(f"  [TD3 Eval] avg_reward={avg_r:.3f} | {breakdown_str}")
    return avg_r

# -----------------------------------------------------------------------------
# Plotting Tools (Retaining high-quality visualization)
# -----------------------------------------------------------------------------


def save_training_plots(save_dir, actor_loss_log, critic_loss_log, eval_steps_log, eval_reward_log, stage_switch_steps):
    """Plots actor/critic loss and eval reward, saving them to save_dir."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Training Curves", fontsize=14)

    # -- Top Plot: actor / critic loss ----------------------------------------
    ax = axes[0]
    if actor_loss_log:
        steps_a, vals_a = zip(*actor_loss_log)
        ax.plot(steps_a, vals_a, label="actor_loss",
                color="steelblue", linewidth=1)
    if critic_loss_log:
        steps_c, vals_c = zip(*critic_loss_log)
        ax2 = ax.twinx()
        ax2.plot(steps_c, vals_c, label="critic_loss",
                 color="tomato",    linewidth=1, linestyle="--")
        ax2.set_ylabel("critic_loss", color="tomato")
        ax2.tick_params(axis="y", labelcolor="tomato")
        ax2.legend(loc="upper right")
    for s, st in stage_switch_steps:
        ax.axvline(x=s, color="gray", linestyle=":", linewidth=0.8)
        ax.text(s, ax.get_ylim()[1], f"s{st}",
                fontsize=7, color="gray", va="top")
    ax.set_xlabel("steps")
    ax.set_ylabel("actor_loss", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # -- Bottom Plot: eval reward ---------------------------------------------
    ax = axes[1]
    if eval_reward_log:
        ax.plot(eval_steps_log, eval_reward_log, marker="o", markersize=3,
                color="seagreen", linewidth=1.2, label="eval_avg_reward")
        # Highlight the best point
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
    print(f"[plot] Training curves saved to {out_path}")


def main(args):
    # 1. Environment Initialization
    cfg = EnvConfig(headless=True, max_steps=args.max_episode_steps)
    env = HoleBoardEnv(cfg, sim_app=sim_app)
    env.set_stage(args.stage)

    # Dimension assignment
    args.state_dim, args.priv_dim, args.action_dim = SCALAR_DIM, PRIVILEGED_DIM, ACTION_DIM
    args.img_h, args.img_w, args.max_action = IMG_H, IMG_W, 1.0

    # 2. TD3 Component Initialization
    run_id = time.strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"{args.save_dir}/TD3_{run_id}"
    save_dir = f"{base_save_dir}/stage{args.stage}"
    os.makedirs(save_dir, exist_ok=True)

    replay_buffer = ReplayBuffer(args)
    agent = TD3(args)
    writer = SummaryWriter(log_dir=f"TD3/runs/stage{args.stage}/TD3_{run_id}")

    if args.resume:
        agent.load(args.resume)

    reward_norm = Normalization(shape=1) if args.use_reward_norm else None

    total_steps, episode_idx, evaluate_rewards = 0, 0, []
    stage = env.cfg.stage
    actor_loss_log, critic_loss_log, eval_steps_log, eval_reward_log, stage_switch_steps = [], [], [], [], []

    print(
        f"TD3 Training Started | Warmup Steps={args.random_steps} | Device={agent.device}")

    # 3. Main Training Loop
    while total_steps < args.max_train_steps:
        episode_idx += 1
        scalar_obs, priv_obs, wrist_d, global_d, _ = env.reset()

        # Automatic curriculum stage switching detection
        if env.cfg.stage != stage:
            # 1. Save final weights for the previous stage
            agent.save(f"{save_dir}/final_stage{stage}.pt",
                       stage, env.scalar_norm, env.priv_norm)

            # 2. Update to new stage
            stage = env.cfg.stage

            # 3. Dynamically switch save path and create new folder
            save_dir = f"{base_save_dir}/stage{stage}"
            os.makedirs(save_dir, exist_ok=True)

            # 4. CRITICAL: Clear previous evaluation scores to recalculate best.pt for new stage
            evaluate_rewards = []

            stage_switch_steps.append((total_steps, stage))
            print(
                f"[Curriculum] TD3 Entering Stage {stage} | New weights will be saved at: {save_dir}")

        done, ep_steps = False, 0
        while not done:
            ep_steps += 1

            # --- Action Sampling ---
            if total_steps < args.random_steps:
                # Warmup phase: Completely random exploration
                action = np.random.uniform(-1.0, 1.0,
                                           size=ACTION_DIM).astype(np.float32)
            else:
                # Training phase: Actor output + Exploration noise
                action = agent.select_action(scalar_obs, wrist_d, global_d)
                action = (action + np.random.normal(0, args.expl_noise,
                          size=ACTION_DIM)).clip(-1.0, 1.0)

            # --- Environment Interaction ---
            scalar_obs_, priv_obs_, wrist_d_, global_d_, r, done, info = env.step(
                action)

            if reward_norm:
                r = float(reward_norm(np.array([r]))[0])
            dw = done and not info.get("timeout", False)

            # --- Store Memory (No a_logprob stored) ---
            replay_buffer.store(scalar_obs, priv_obs, wrist_d, global_d, action, r,
                                scalar_obs_, priv_obs_, wrist_d_, global_d_, dw, done)

            scalar_obs, priv_obs, wrist_d, global_d = scalar_obs_, priv_obs_, wrist_d_, global_d_
            total_steps += 1

            # --- Network Update ---
            if total_steps >= args.random_steps:
                metrics = agent.update(replay_buffer, args.batch_size)

                if total_steps % 100 == 0:
                    writer.add_scalar("train/critic_loss",
                                      metrics["critic_loss"], total_steps)
                    critic_loss_log.append(
                        (total_steps, metrics["critic_loss"]))
                    if metrics["actor_loss"] != 0:
                        writer.add_scalar("train/actor_loss",
                                          metrics["actor_loss"], total_steps)
                        actor_loss_log.append(
                            (total_steps, metrics["actor_loss"]))

                # Save regular checkpoints every 5000 steps
                if total_steps > 0 and total_steps % 5000 == 0:
                    agent.save(f"{save_dir}/ckpt_{total_steps}.pt",
                               stage, env.scalar_norm, env.priv_norm)
                    print(
                        f"\n[Checkpoint] Mandatory weight save at {total_steps} steps: ckpt_{total_steps}.pt\n")

        # --- Periodic Evaluation ---
        if total_steps % args.evaluate_freq < args.max_episode_steps and total_steps > args.random_steps:
            avg_r = evaluate_policy(env, agent, n_episodes=10, stage=stage)
            evaluate_rewards.append(avg_r)
            eval_steps_log.append(total_steps)
            eval_reward_log.append(avg_r)
            writer.add_scalar("eval/reward", avg_r, total_steps)

            # Auto-generate training curves after each evaluation
            save_training_plots(save_dir, actor_loss_log, critic_loss_log,
                                eval_steps_log, eval_reward_log, stage_switch_steps)

            if avg_r >= max(evaluate_rewards):
                agent.save(f"{save_dir}/best.pt", stage,
                           env.scalar_norm, env.priv_norm)
                print(
                    f"\n[Best] New high score {avg_r:.2f}! Updated best.pt weight file!\n")

    agent.save(f"{save_dir}/final.pt", stage, env.scalar_norm, env.priv_norm)
    env.close()
    print("TD3 Training Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TD3 Training")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--random_steps", type=int,
                        default=10000, help="Warmup steps")
    parser.add_argument("--expl_noise", type=float,
                        default=0.1, help="Exploration Gaussian noise")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=int(1e5))
    parser.add_argument("--lr_a", type=float, default=1e-4)
    parser.add_argument("--lr_c", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--max_train_steps", type=int, default=int(1e6))
    parser.add_argument("--evaluate_freq", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="TD3/checkpoints")
    parser.add_argument("--max_episode_steps", type=int, default=400)
    parser.add_argument("--use_reward_norm", type=bool, default=True)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()
    main(args)
