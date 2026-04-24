from isaacsim import SimulationApp
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SimulationApp must be started before Isaac-related imports


def parse_args():
    parser = argparse.ArgumentParser("SAC Actor-only evaluation")
    parser.add_argument("--checkpoint", type=str,
                        required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--stage", type=int, default=0,
                        help="Curriculum stage 1/2/3, 0 means auto-read from checkpoint")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of test episodes")
    parser.add_argument("--max_episode_steps", type=int,
                        default=400, help="Maximum steps per episode")
    parser.add_argument("--render-delay", type=float, default=0.02,
                        dest="render_delay", help="Rendering delay per step (seconds)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true",
                        help="Headless mode (disable rendering window)")
    return parser.parse_args()


def restore_norm(norm_obj, stats: dict):
    norm_obj.running_ms.n = stats["n"]
    norm_obj.running_ms.mean = stats["mean"].copy()
    norm_obj.running_ms.S = stats["S"].copy()
    norm_obj.running_ms.std = stats["std"].copy()


def load_actor(checkpoint_path, device):
    import torch
    from SAC.network import SACActor

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Automatically infer hidden_dim from weights to avoid inconsistency between eval script and training hyperparams
    hidden_dim = int(ckpt["actor"]["mean_head.weight"].shape[1])
    actor = SACActor(hidden_dim=hidden_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    stage = int(ckpt.get("stage", 1))
    n_param = sum(p.numel() for p in actor.parameters())

    print(f"[INFO] Loaded weights: {checkpoint_path}")
    print(f"[INFO] Curriculum Stage: stage{stage}")
    print(f"[INFO] Actor hidden_dim: {hidden_dim}")
    print(f"[INFO] Actor Parameter Count: {n_param:,}")

    obs_norm_stats = None
    priv_norm_stats = None
    if "obs_norm_n" in ckpt:
        obs_norm_stats = {
            "n": ckpt["obs_norm_n"],
            "mean": ckpt["obs_norm_mean"],
            "S": ckpt["obs_norm_S"],
            "std": ckpt["obs_norm_std"],
        }
        print("[INFO] obs_norm statistics loaded")
    else:
        print("[WARN] Checkpoint does not contain obs_norm statistics, observation normalization may be incorrect")

    if "priv_norm_n" in ckpt:
        priv_norm_stats = {
            "n": ckpt["priv_norm_n"],
            "mean": ckpt["priv_norm_mean"],
            "S": ckpt["priv_norm_S"],
            "std": ckpt["priv_norm_std"],
        }

    return actor, stage, obs_norm_stats, priv_norm_stats


def get_action(actor, scalar_obs, wrist_d, global_d, device):
    import torch

    s = torch.tensor(scalar_obs, dtype=torch.float32,
                     device=device).unsqueeze(0)
    wd = torch.tensor(wrist_d, dtype=torch.float32,
                      device=device).unsqueeze(0).unsqueeze(0)
    gd = torch.tensor(global_d, dtype=torch.float32,
                      device=device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        action, _ = actor.sample(
            s, wd, gd, deterministic=True, with_logprob=False)
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
    stage = args.stage if args.stage in (1, 2, 3, 4) else ckpt_stage
    if stage not in (1, 2, 3, 4):
        stage = 1

    cfg = EnvConfig(headless=args.headless, max_steps=args.max_episode_steps)
    env = HoleBoardEnv(cfg, sim_app=sim_app)
    env.set_stage(stage)

    if obs_norm_stats is not None:
        restore_norm(env.scalar_norm, obs_norm_stats)
    if priv_norm_stats is not None:
        restore_norm(env.priv_norm, priv_norm_stats)

    env.training = False

    print(
        f"\n=== Starting Evaluation | stage={stage} | episodes={args.episodes} | render_delay={args.render_delay}s ===\n")

    ep_steps_all = []
    ep_dist_all = []
    ep_coll_cnt = []
    success_count = 0
    coll_term_count = 0
    timeout_count = 0
    success_steps = []

    for ep in range(1, args.episodes + 1):
        scalar_obs, _priv, wrist_d, global_d, _ = env.reset()
        done = False
        steps = 0
        coll_steps = 0

        while not done:
            action = get_action(actor, scalar_obs, wrist_d, global_d, device)
            scalar_obs, _priv, wrist_d, global_d, _reward, done, info = env.step(
                action)
            steps += 1

            if env.reward_breakdown.get("collision", 0.0) < 0.0:
                coll_steps += 1

            if args.render_delay > 0:
                time.sleep(args.render_delay)

        dist = info.get("dist_to_target", float("nan"))
        is_success = bool(info.get("success", False))
        is_coll_end = bool(info.get("collision", False))
        is_timeout = bool(info.get("timeout", False))

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

    n = args.episodes
    avg_steps = sum(ep_steps_all) / n
    avg_dist = sum(ep_dist_all) / n
    avg_coll = sum(ep_coll_cnt) / n
    avg_suc_steps = sum(success_steps) / max(1, len(success_steps))

    print()
    print("Metric                         Value")
    print("────────────────────────────────")
    print(f"Total Test Episodes           {n}")
    print(f"Success Rate                  {success_count / n * 100:.1f}%")
    print(f"Collision Termination Rate    {coll_term_count / n * 100:.1f}%")
    print(f"Timeout Rate                  {timeout_count / n * 100:.1f}%")
    print(f"Average Episode Steps         {avg_steps:.1f}")
    print(f"Average Successful Steps      {avg_suc_steps:.1f}")
    print(f"Avg Final Dist (to Target)    {avg_dist:.3f} m")
    print(f"Avg Cumulative Collisions/ep  {avg_coll:.2f}")
    print("────────────────────────────────")
    print(f"Best Episode Steps            {min(ep_steps_all)}")
    print(f"Worst Episode Steps           {max(ep_steps_all)}")

    env.close()


if __name__ == "__main__":
    main()
