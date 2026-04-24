from isaacsim import SimulationApp
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SimulationApp must start before importing IsaacSim-dependent modules.


def parse_args():
    parser = argparse.ArgumentParser("Actor-only evaluation")
    parser.add_argument("--checkpoint",         type=str,
                        required=True,   help="Path to the checkpoint .pt file")
    parser.add_argument("--stage",              type=int,   default=0,
                        help="Curriculum stage 1/2/3, 0 means auto-read from checkpoint")
    parser.add_argument("--episodes",           type=int,
                        default=10,      help="Number of test episodes")
    parser.add_argument("--max_episode_steps",  type=int,
                        default=400,     help="Maximum steps per episode")
    parser.add_argument("--render-delay",       type=float, default=0.02,
                        dest="render_delay", help="Rendering delay per step (seconds), default 0.02s/step")
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--headless",           action="store_true",
                        help="Headless mode (disables the rendering window)")
    return parser.parse_args()


def load_actor(checkpoint_path, device):
    """Loads only Actor network weights, ignoring Critic and optimizer.
    Also returns obs_norm / priv_norm statistics dictionaries if present in the checkpoint.
    """
    import torch
    from PPO.network import ActorNetwork

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor = ActorNetwork().to(device)
    missing, unexpected = actor.load_state_dict(ckpt["actor"], strict=False)

    if missing:
        print(
            f"[WARN] Missing keys (will use default initialization): {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys (ignored): {unexpected}")

    actor.eval()

    stage = int(ckpt.get("stage", 1))
    n_param = sum(p.numel() for p in actor.parameters())

    print(f"[INFO] Weights loaded from: {checkpoint_path}")
    print(f"[INFO] Training steps: N/A (Steps not saved in checkpoint)")
    print(f"[INFO] Curriculum Stage: stage{stage}")
    print(f"[INFO] Actor Parameters: {n_param:,}")

    obs_norm_stats = None
    priv_norm_stats = None

    # Load normalization statistics to prevent observation scale mismatch
    if "obs_norm_n" in ckpt:
        obs_norm_stats = {
            "n":    ckpt["obs_norm_n"],
            "mean": ckpt["obs_norm_mean"],
            "S":    ckpt["obs_norm_S"],
            "std":  ckpt["obs_norm_std"],
        }
        print("[INFO] obs_norm statistics loaded.")
    else:
        print(
            "[WARN] Checkpoint does not contain obs_norm stats. Normalization might be incorrect!")

    if "priv_norm_n" in ckpt:
        priv_norm_stats = {
            "n":    ckpt["priv_norm_n"],
            "mean": ckpt["priv_norm_mean"],
            "S":    ckpt["priv_norm_S"],
            "std":  ckpt["priv_norm_std"],
        }

    return actor, stage, obs_norm_stats, priv_norm_stats


def restore_norm(norm_obj, stats: dict):
    """Injects saved statistics into the Normalization object."""
    norm_obj.running_ms.n = stats["n"]
    norm_obj.running_ms.mean = stats["mean"].copy()
    norm_obj.running_ms.S = stats["S"].copy()
    norm_obj.running_ms.std = stats["std"].copy()


def get_action(actor, scalar_obs, wrist_d, global_d, device):
    """Deterministic inference: Uses the mean of the Gaussian distribution (no sampling)."""
    import torch
    s = torch.tensor(scalar_obs, dtype=torch.float).unsqueeze(0).to(device)
    wd = torch.tensor(wrist_d,   dtype=torch.float).unsqueeze(
        0).unsqueeze(0).to(device)
    gd = torch.tensor(global_d,  dtype=torch.float).unsqueeze(
        0).unsqueeze(0).to(device)

    with torch.no_grad():
        action = actor(s, wd, gd)   # forward() returns tanh(mean) directly
    return action.cpu().numpy().flatten()


def main():
    args = parse_args()
    sim_app = SimulationApp(
        {"headless": args.headless, "renderer": "RayTracedLighting"})

    import numpy as np
    import torch
    from common.env import HoleBoardEnv, EnvConfig

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor, ckpt_stage, obs_norm_stats, priv_norm_stats = load_actor(
        args.checkpoint, device)

    # Priority: Command line arg > Checkpoint value
    stage = args.stage if args.stage in (1, 2, 3) else ckpt_stage
    if stage not in (1, 2, 3):
        stage = 1

    cfg = EnvConfig(headless=args.headless, max_steps=args.max_episode_steps)
    env = HoleBoardEnv(cfg, sim_app=sim_app)
    env.set_stage(stage)

    # Restore training-time normalization stats to avoid obs explosion due to std≈0
    if obs_norm_stats is not None:
        restore_norm(env.scalar_norm, obs_norm_stats)
    if priv_norm_stats is not None:
        restore_norm(env.priv_norm, priv_norm_stats)

    env.training = False   # Freeze normalizer stats during evaluation

    print(
        f"\n=== Evaluation Start | stage={stage} | episodes={args.episodes} | render_delay={args.render_delay}s ===\n")

    # -- Statistics Containers -------------------------------------------------
    ep_steps_all = []
    ep_dist_all = []
    ep_coll_cnt = []   # Cumulative collision steps per episode
    success_count = 0
    coll_term_count = 0
    timeout_count = 0
    success_steps = []   # Steps for successful episodes only

    # -- Main Loop -------------------------------------------------------------
    for ep in range(1, args.episodes + 1):
        scalar_obs, _, wrist_d, global_d, _ = env.reset()
        done = False
        steps = 0
        coll_steps = 0

        while not done:
            action = get_action(actor, scalar_obs, wrist_d, global_d, device)
            scalar_obs, _, wrist_d, global_d, _reward, done, info = env.step(
                action)
            steps += 1

            # Check for soft collisions via reward breakdown
            if env.reward_breakdown.get("collision", 0.0) < 0.0:
                coll_steps += 1

            if args.render_delay > 0:
                time.sleep(args.render_delay)

        dist = info.get("dist_to_target", float("nan"))
        is_success = bool(info.get("success",   False))
        is_coll_end = bool(info.get("collision",  False))
        is_timeout = bool(info.get("timeout",    False))

        ep_steps_all.append(steps)
        ep_dist_all.append(dist)
        ep_coll_cnt.append(coll_steps)

        if is_success:
            success_count += 1
            success_steps.append(steps)
        if is_coll_end:
            coll_term_count += 1
        if is_timeout:
            timeout_count += 1

        result_tag = "SUCCESS" if is_success else (
            "COLLISION" if is_coll_end else "TIMEOUT")
        print(
            f"[ep {ep:2d}/{args.episodes}]  {result_tag}  |  steps={steps:3d}  |  dist_to_target={dist:.3f} m")

    # -- Summary Statistics ----------------------------------------------------
    n = args.episodes
    avg_steps = sum(ep_steps_all) / n
    avg_dist = sum(ep_dist_all) / n
    avg_coll = sum(ep_coll_cnt) / n
    avg_suc_steps = sum(success_steps) / max(1, len(success_steps))

    print("\n" + "Metric".ljust(30) + "Value")
    print("─" * 45)
    print(f"Total Test Episodes:".ljust(30) + f"{n}")
    print(f"Success Rate:".ljust(30) + f"{success_count / n * 100:.1f}%")
    print(f"Collision Rate (Terminal):".ljust(
        30) + f"{coll_term_count / n * 100:.1f}%")
    print(f"Timeout Rate:".ljust(30) + f"{timeout_count / n * 100:.1f}%")
    print(f"Average Episode Steps:".ljust(30) + f"{avg_steps:.1f}")
    print(f"Average Success Steps:".ljust(30) + f"{avg_suc_steps:.1f}")
    print(f"Average Final Dist to Target:".ljust(30) + f"{avg_dist:.3f} m")
    print(f"Average Collisions per Ep:".ljust(30) + f"{avg_coll:.2f}")
    print("─" * 45)
    print(f"Min Steps (Best Episode):".ljust(30) + f"{min(ep_steps_all)}")
    print(f"Max Steps (Worst Episode):".ljust(30) + f"{max(ep_steps_all)}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
