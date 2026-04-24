# -- Add parent directory to sys.path so me5406_project package can be imported --
from common.normalization import Normalization, RewardScaling
from PPO.replaybuffer import ReplayBuffer
from PPO.ppo import PPO_continuous
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
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# !! SimulationApp must be started before all other imports !!
sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

matplotlib.use("Agg")   # Use non-interactive backend for headless environments


# -----------------------------------------------------------------------------
# Policy Evaluation
# (Disables normalization updates, uses deterministic mean actions)
# Used for periodic assessment and printing reward breakdowns.
# -----------------------------------------------------------------------------
def evaluate_policy(env, agent, n_episodes=3, stage=1):
    """Runs n_episodes of evaluation, returns avg reward, and prints breakdown."""
    env.training = False     # Freeze normalization statistics
    total_reward = 0.0
    term_keys = ["progress", "target_guide", "align", "post_align",
                 "pass", "collision", "smooth", "arrive", "alive"]
    term_sums = {k: 0.0 for k in term_keys}
    total_steps = 0

    for _ in range(n_episodes):
        scalar_obs, priv_obs, wrist_d, global_d, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.evaluate(scalar_obs, wrist_d, global_d, stage)
            scalar_obs, priv_obs, wrist_d, global_d, r, done, _ = env.step(
                action)
            ep_reward += r
            bd = env.reward_breakdown
            for k in term_keys:
                term_sums[k] += bd.get(k, 0.0)
            total_steps += 1
        total_reward += ep_reward

    env.training = True     # Resume training mode
    avg_r = total_reward / n_episodes
    if total_steps > 0:
        # Average each component per episode to align with avg_reward
        avg_terms = {k: term_sums[k] / n_episodes for k in term_keys}
        env._reward_manager.last_breakdown = avg_terms.copy()
        breakdown_str = "  ".join(
            f"{k}={v:+.4f}" for k, v in avg_terms.items())
        print(f"  [breakdown/eval_avg] {breakdown_str}")
    return avg_r


# -----------------------------------------------------------------------------
# Plotting Utility
# -----------------------------------------------------------------------------
def save_training_plots(save_dir, actor_loss_log, critic_loss_log, eval_steps_log, eval_reward_log, stage_switch_steps):
    """Plots actor/critic loss and eval reward, saving the result to save_dir."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Training Curves", fontsize=14)

    # -- Top Plot: Actor / Critic Loss --
    ax = axes[0]
    if actor_loss_log:
        steps_a, vals_a = zip(*actor_loss_log)
        ax.plot(steps_a, vals_a, label="actor_loss",
                color="steelblue", linewidth=1)
    if critic_loss_log:
        steps_c, vals_c = zip(*critic_loss_log)
        ax2 = ax.twinx()
        ax2.plot(steps_c, vals_c, label="critic_loss",
                 color="tomato", linewidth=1, linestyle="--")
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

    # -- Bottom Plot: Eval Reward --
    ax = axes[1]
    if eval_reward_log:
        ax.plot(eval_steps_log, eval_reward_log, marker="o", markersize=3,
                color="seagreen", linewidth=1.2, label="eval_avg_reward")
        # Highlight best point
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
    # -- Environment Initialization --
    cfg = EnvConfig(
        headless=True,
        max_steps=args.max_episode_steps,
        debug_diagnostics=args.debug_diagnostics,
        debug_log_episodes=args.debug_log_episodes,
        debug_log_steps=args.debug_log_steps,
    )
    env = HoleBoardEnv(cfg, sim_app=sim_app)
    env.set_stage(args.stage)

    # -- Append Dimension Info for ReplayBuffer --
    args.state_dim = SCALAR_DIM
    args.priv_dim = PRIVILEGED_DIM
    args.action_dim = ACTION_DIM
    args.img_h = IMG_H
    args.img_w = IMG_W
    args.max_action = 1.0   # Actions scaled inside env; PPO side stays [-1, 1]

    # -- Component Initialization --
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.save_dir}/stage{args.stage}/{run_id}"
    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)
    writer = SummaryWriter(log_dir=f"runs/stage{args.stage}/{run_id}")

    if args.resume:
        # reset_optimizer=True: Keep network weights but clear Adam momentum/adv_norm stats.
        # Prevents momentum from old reward structures causing strategy collapse.
        prev_stage = agent.load(args.resume, reset_optimizer=True)
        print(
            f"[resume] Loaded model from {args.resume} (last stage={prev_stage}), optimizer state cleared")
        if prev_stage != args.stage:
            agent.reset_lr()
            print(
                f"[stage switch] {prev_stage} → {args.stage}, LR has been reset")

    # Reward Normalization / Scaling (Prioritize scaling)
    reward_norm = Normalization(shape=1) if args.use_reward_norm else None
    reward_scaling = RewardScaling(
        gamma=args.gamma) if args.use_reward_scaling else None

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    total_steps = 0
    episode_idx = 0
    evaluate_num = 0
    last_eval_steps = 0
    # Rolling record of recent 5 training rewards
    recent_rewards = deque(maxlen=5)
    evaluate_rewards = []
    stage = env.cfg.stage    # Track current stage (for auto-curriculum)
    # Episodes remaining for tightened KL after stage switch
    kl_tight_episodes = 0

    # -- Log Data Collection --
    actor_loss_log = []
    critic_loss_log = []
    eval_steps_log = []
    eval_reward_log = []
    stage_switch_steps = []

    if args.stage > 1:
        kl_tight_episodes = 20
        agent.target_kl = 0.01
        print(
            f"[kl] Initial stage={args.stage}, target_kl tightened to 0.01 for {kl_tight_episodes} episodes")

    print(f"Training Start | stage={stage} | max_steps={args.max_train_steps}")

    # -- Main Training Loop --
    debug_episode = 0
    while total_steps < args.max_train_steps:
        episode_idx += 1
        scalar_obs, priv_obs, wrist_d, global_d, _ = env.reset()

        # env.reset() applies pending_stage; check if auto-curriculum triggered
        new_stage = env.cfg.stage
        if new_stage != stage:
            print(
                f"[curriculum] stage {stage} → {new_stage}, resetting optimizer momentum")
            agent.reset_lr()
            stage = new_stage
            stage_switch_steps.append((total_steps, stage))
            kl_tight_episodes = 10
            agent.target_kl = 0.01
            if reward_scaling is not None:
                reward_scaling.running_ms.n = 0
                reward_scaling.running_ms.mean = np.zeros(1)
                reward_scaling.running_ms.S = np.zeros(1)
                reward_scaling.running_ms.std = np.ones(1)
                reward_scaling.R = np.zeros(1)
                print(f"[reward_scaling] Stage switch: normalization stats reset")
            print(
                f"[kl] Stage switch: target_kl tightened to 0.01 for {kl_tight_episodes} episodes")

        episode_steps = 0
        episode_reward = 0.0
        done = False
        if args.debug_diagnostics:
            debug_episode += 1
            debug_actions = []

        while not done:
            episode_steps += 1

            # 1. Sample Action
            action, a_logprob = agent.interact(
                scalar_obs, wrist_d, global_d, stage=stage)

            # 2. Step Environment
            scalar_obs_, priv_obs_, wrist_d_, global_d_, r, done, info = env.step(
                action)
            episode_reward += r

            # 3. Reward Processing (Scaling preferred over Norm)
            if reward_scaling is not None:
                r = reward_scaling(r)
            elif reward_norm is not None:
                r = float(reward_norm(np.array([r]))[0])

            # 4. Check dw (True death/success vs Timeout)
            dw = done and not info.get("timeout", False)

            # 5. Store in Buffer
            replay_buffer.store(
                scalar_obs, priv_obs, wrist_d,  global_d,
                action, a_logprob, r,
                scalar_obs_, priv_obs_, wrist_d_, global_d_,
                dw, done,
            )

            scalar_obs = scalar_obs_
            priv_obs = priv_obs_
            wrist_d = wrist_d_
            global_d = global_d_
            total_steps += 1

            if args.debug_diagnostics and debug_episode <= args.debug_log_episodes:
                debug_actions.append(action.copy())

            # 6. Update when buffer is full
            if replay_buffer.count == args.batch_size:
                metrics = agent.update(replay_buffer, total_steps, stage=stage)
                replay_buffer.count = 0
                if args.use_lr_decay:
                    agent.lr_decay(total_steps)
                print(f"[update] steps={total_steps:>7d} | "
                      f"actor_loss={metrics['actor_loss']:+.4f} | "
                      f"critic_loss={metrics['critic_loss']:.4f} | "
                      f"aux_loss={metrics['aux_loss']:.4f} | "
                      f"wrist_std={metrics['wrist_feat_std']:.3f} | "
                      f"global_std={metrics['global_feat_std']:.3f} | "
                      f"wrist_gnorm={metrics['wrist_grad_norm']:.3f} | "
                      f"global_gnorm={metrics['global_grad_norm']:.3f}")

                writer.add_scalar("train/actor_loss",
                                  metrics["actor_loss"], total_steps)
                writer.add_scalar("train/critic_loss",
                                  metrics["critic_loss"], total_steps)
                writer.add_scalar("train/aux_loss",
                                  metrics["aux_loss"], total_steps)
                writer.add_scalar("cnn/wrist_feat_std",
                                  metrics["wrist_feat_std"], total_steps)
                writer.add_scalar("cnn/global_feat_std",
                                  metrics["global_feat_std"], total_steps)
                writer.add_scalar("cnn/wrist_grad_norm",
                                  metrics["wrist_grad_norm"], total_steps)
                writer.add_scalar("cnn/global_grad_norm",
                                  metrics["global_grad_norm"], total_steps)

                actor_loss_log.append((total_steps, metrics["actor_loss"]))
                critic_loss_log.append((total_steps, metrics["critic_loss"]))

                if total_steps % args.save_interval < args.batch_size:
                    ckpt_path = f"{save_dir}/ckpt_{total_steps}.pt"
                    agent.save(ckpt_path, stage=stage,
                               obs_norm=env.scalar_norm, priv_norm=env.priv_norm)
                    print(f"[checkpoint] Saved {ckpt_path}")

        # Episode end: cleanup reward scaling and KL settings
        if kl_tight_episodes > 0:
            kl_tight_episodes -= 1
            if kl_tight_episodes == 0:
                agent.target_kl = args.target_kl
                print(
                    f"[kl] Tightened period over, target_kl restored to {args.target_kl}")

        recent_rewards.append(episode_reward)
        if len(recent_rewards) == recent_rewards.maxlen:
            writer.add_scalar(
                "train/reward_avg5", sum(recent_rewards) / len(recent_rewards), total_steps)

        print(
            f"[stage {stage}][episode {episode_idx}] done | steps={episode_steps}")
        if reward_scaling is not None:
            reward_scaling.reset()

        # -- Periodic Evaluation --
        if total_steps - last_eval_steps >= args.evaluate_freq:
            evaluate_num += 1
            avg_r = evaluate_policy(env, agent, n_episodes=3, stage=stage)
            evaluate_rewards.append(avg_r)
            eval_steps_log.append(total_steps)
            eval_reward_log.append(avg_r)
            writer.add_scalar("eval/episode_reward", avg_r, total_steps)
            print(
                f"[eval {evaluate_num}] steps={total_steps}  avg_reward={avg_r:.3f}")
            last_eval_steps = total_steps
            save_training_plots(save_dir, actor_loss_log, critic_loss_log,
                                eval_steps_log, eval_reward_log, stage_switch_steps)

            if avg_r >= max(evaluate_rewards):
                best_path = f"{save_dir}/best.pt"
                agent.save(best_path, stage=stage,
                           obs_norm=env.scalar_norm, priv_norm=env.priv_norm)
                print(f"[best] avg_r={avg_r:.3f}, saved {best_path}")

        if args.debug_diagnostics and debug_episode <= args.debug_log_episodes:
            if len(debug_actions) > 0:
                a = np.stack(debug_actions, axis=0)
                a_mean = float(a.mean())
                a_std = float(a.std())
                a_abs = float(np.mean(np.abs(a)))
                a_sat = float(np.mean(np.abs(a) >= (args.max_action - 1e-6)))
                print(
                    "[debug][episode] ep=%d steps=%d a_mean=%.4f a_std=%.4f a_abs=%.4f a_sat=%.3f"
                    % (debug_episode, episode_steps, a_mean, a_std, a_abs, a_sat)
                )

    final_path = f"{save_dir}/final.pt"
    agent.save(final_path, stage=stage,
               obs_norm=env.scalar_norm, priv_norm=env.priv_norm)
    print(f"[final] Saved {final_path}")
    save_training_plots(save_dir, actor_loss_log, critic_loss_log,
                        eval_steps_log, eval_reward_log, stage_switch_steps)
    env.close()
    writer.close()
    print("Training finished")


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("ME5406 PPO Training")

    # Save / Resume
    parser.add_argument("--save_dir",      type=str, default="checkpoints")
    parser.add_argument("--save_interval", type=int,
                        default=10000, help="Steps between checkpoint saves")
    parser.add_argument("--resume",        type=str,
                        default="",    help="Path to resume training from")

    # Training Scale
    parser.add_argument("--max_train_steps",  type=int,   default=int(1.5e6))
    parser.add_argument("--max_episode_steps", type=int,   default=400)
    parser.add_argument("--evaluate_freq",    type=int,
                        default=2048, help="Steps between evaluations")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--stage",            type=int,
                        default=1,    help="Initial curriculum stage 1/2/3")

    # PPO Hyperparameters
    parser.add_argument("--batch_size",       type=int,   default=2048)
    parser.add_argument("--mini_batch_size",  type=int,   default=32)
    parser.add_argument("--K_epochs",         type=int,   default=5)
    parser.add_argument("--lr_a",             type=float, default=2e-4)
    parser.add_argument("--lr_c",             type=float, default=2e-4)
    parser.add_argument("--gamma",            type=float, default=0.99)
    parser.add_argument("--lamda",            type=float, default=0.95)
    parser.add_argument("--epsilon",          type=float, default=0.2)
    parser.add_argument("--entropy_coef",     type=float, default=0.02)
    parser.add_argument("--target_kl",        type=float,
                        default=0.02,  help="KL threshold for early stopping")

    # Tricks
    parser.add_argument("--use_adv_norm",       type=bool,  default=True)
    parser.add_argument("--use_reward_norm",    type=bool,  default=False)
    parser.add_argument("--use_reward_scaling", type=bool,  default=True)
    parser.add_argument("--use_lr_decay",       type=bool,  default=True)
    parser.add_argument("--use_grad_clip",      type=bool,  default=True)
    parser.add_argument("--set_adam_eps",       type=bool,  default=True)
    parser.add_argument("--lr_parameter",       type=float,
                        default=0.95, help="LR recovery parameter")
    parser.add_argument("--aux_coef",           type=float,
                        default=0.3,  help="Weight for wrist CNN aux loss")

    # Debug
    parser.add_argument("--debug_diagnostics",  type=bool, default=False)
    parser.add_argument("--debug_log_episodes", type=int,  default=3)
    parser.add_argument("--debug_log_steps",    type=int,  default=5)

    args = parser.parse_args()
    main(args)
