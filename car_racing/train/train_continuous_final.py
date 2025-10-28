import os
import csv
import time
import gymnasium as gym
import torch
import numpy as np

from car_racing.wrappers.grayscale_wrapper import GrayScaleWrapper
from car_racing.wrappers.frame_stack import FrameStack
from car_racing.models.cnn_policy_value_cont import CnnPolicyValueCont
from car_racing.agents.ppo_agent_cont import PPOAgentCont
from car_racing.utils.plot import save_reward_curve
from car_racing.utils.live_plot import LivePlotter  # <-- NEW


#################################################################
# 1. ENV FACTORY
#################################################################

def make_env_continuous(render_mode=None, domain_randomize=False):
    base_env = gym.make(
        "CarRacing-v3",
        render_mode=render_mode,
        continuous=True,
        domain_randomize=domain_randomize,
        lap_complete_percent=0.95,
    )
    env = GrayScaleWrapper(base_env)
    env = FrameStack(env, k=4)
    return env


#################################################################
# 2. 4-PHASE SCHEDULER (lr / entropy / horizon)
#################################################################

TOTAL_STEPS = 6_000_000

PHASES = [
    (0,          1_500_000, 5e-5,   5e-5,   1e-3, 1e-3,   4096, 4096),
    (1_500_000,  3_000_000, 5e-5,   3e-5,   1e-3, 7e-4,   4096, 3072),
    (3_000_000,  4_500_000, 3e-5,   1.5e-5, 7e-4, 4e-4,   3072, 2048),
    (4_500_000,  6_000_000, 1.5e-5, 5e-6,   4e-4, 1e-4,   2048, 1024),
]

def _interp(start_val, end_val, ratio):
    return start_val + ratio * (end_val - start_val)

def _get_phase_and_ratio(step: int):
    for i, (p_start, p_end, *_rest) in enumerate(PHASES):
        if step < p_end or i == len(PHASES) - 1:
            phase_len = p_end - p_start
            if phase_len <= 0:
                ratio = 1.0
            else:
                ratio = np.clip((step - p_start) / phase_len, 0.0, 1.0)
            return i, ratio
    return len(PHASES) - 1, 1.0

def lr_schedule(step: int) -> float:
    phase_idx, ratio = _get_phase_and_ratio(step)
    (_s, _e,
     lr_start, lr_end,
     _ent_s, _ent_e,
     _hor_s, _hor_e) = PHASES[phase_idx]
    return float(_interp(lr_start, lr_end, ratio))

def entropy_schedule(step: int) -> float:
    phase_idx, ratio = _get_phase_and_ratio(step)
    (_s, _e,
     _lr_s, _lr_e,
     ent_start, ent_end,
     _hor_s, _hor_e) = PHASES[phase_idx]
    return float(_interp(ent_start, ent_end, ratio))

def horizon_schedule(step: int) -> int:
    phase_idx, ratio = _get_phase_and_ratio(step)
    (_s, _e,
     _lr_s, _lr_e,
     _ent_s, _ent_e,
     hor_start, hor_end) = PHASES[phase_idx]

    h_raw = _interp(hor_start, hor_end, ratio)
    h_rounded = int(np.round(h_raw / 64.0) * 64)
    return max(h_rounded, 64)


#################################################################
# 3. TRAIN LOOP
#################################################################

def train_continuous_final():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("🚗 FINAL PHASE PPO CONTINUOUS TRAIN START")

    os.makedirs("car_racing/checkpoints", exist_ok=True)
    os.makedirs("car_racing/plots", exist_ok=True)
    os.makedirs("car_racing/logs", exist_ok=True)

    csv_path = "car_racing/logs/final_phase_log.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "global_step",
                "episode_idx",
                "ep_reward",
                "recent_mean_top15of20",
                "lr",
                "entropy_coef",
                "rollout_horizon",
                "phase_idx",
            ])

    
    env = make_env_continuous(render_mode=None, domain_randomize=False)
    obs, info = env.reset()

    model = CnnPolicyValueCont(action_dim=env.action_space.shape[0])
    agent = PPOAgentCont(
        model=model,
        lr=5e-5,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=1e-3,
        update_epochs=4,
        minibatch_size=64,
        device=device,
    )
    
    if not hasattr(agent, "set_lr"):
        def set_lr_fn(new_lr: float):
            for g in agent.optimizer.param_groups:
                g["lr"] = new_lr
        agent.set_lr = set_lr_fn

    
    plotter = LivePlotter()
    rewards_log = []
    recent_mean_log = []
    lr_log = []
    steps_log = []

    # === TRAIN STATE ===
    global_step = 0
    ep_idx = 0
    ep_reward_real = 0.0

    best_mean = -1e9
    last_plot = time.time()

    while global_step < TOTAL_STEPS:

        cur_lr = lr_schedule(global_step)
        cur_ent = entropy_schedule(global_step)
        cur_horz = horizon_schedule(global_step)
        phase_idx, _ratio = _get_phase_and_ratio(global_step)

        agent.set_lr(cur_lr)
        agent.entropy_coef = cur_ent

        buf_obs = []
        buf_actions = []
        buf_logprobs = []
        buf_rewards = []
        buf_dones = []
        buf_values = []

        steps_collected = 0

        while steps_collected < cur_horz and global_step < TOTAL_STEPS:
            # a) action
            action_np, logprob, value = agent.select_action(
                obs,
                deterministic=False
            )

            # b) env.step
            next_obs, reward_env, terminated, truncated, env_info = env.step(action_np)
            done = terminated or truncated

            # c) reward scaling pour PPO
            reward_scaled = reward_env / 100.0

            # d) obs tensor (C,H,W)
            obs_t = torch.from_numpy(obs).float() / 255.0
            obs_t = obs_t.permute(2, 0, 1)

            # e) push buffer
            buf_obs.append(obs_t)
            buf_actions.append(torch.from_numpy(np.array(action_np, dtype=np.float32)))
            buf_logprobs.append(torch.tensor(logprob, dtype=torch.float32))
            buf_rewards.append(torch.tensor(reward_scaled, dtype=torch.float32))
            buf_dones.append(torch.tensor(float(done), dtype=torch.float32))
            buf_values.append(torch.tensor(value, dtype=torch.float32))

            # f) bookkeeping
            ep_reward_real += reward_env
            global_step += 1
            steps_collected += 1
            obs = next_obs

            if done:
                rewards_log.append(ep_reward_real)
                recent_window = rewards_log[-20:]
                trimmed = sorted(recent_window)[max(0, len(recent_window) - 15):]
                recent_mean = (
                    sum(trimmed)/len(trimmed)
                    if len(trimmed) > 0
                    else ep_reward_real
                )
                recent_mean_log.append(recent_mean)

                lr_log.append(cur_lr)
                steps_log.append(global_step)

                print(
                    f"[final4 ep {ep_idx}] "
                    f"rew={ep_reward_real:.2f} "
                    f"recent_mean={recent_mean:.2f} "
                    f"steps={global_step} "
                    f"lr={cur_lr:.2g} "
                    f"ent={cur_ent:.4f} "
                    f"horz={cur_horz} "
                    f"phase={phase_idx}"
                )

                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        global_step,
                        ep_idx,
                        ep_reward_real,
                        recent_mean,
                        cur_lr,
                        cur_ent,
                        cur_horz,
                        phase_idx,
                    ])

                # best checkpoint
                if recent_mean > best_mean:
                    best_mean = recent_mean
                    torch.save(
                        agent.model.state_dict(),
                        "car_racing/checkpoints/ppo_carracing_cont_final_best.pth"
                    )
                    print("💾 Nouveau meilleur modèle -> ppo_carracing_cont_final_best.pth")

                ep_reward_real = 0.0
                ep_idx += 1
                obs, info = env.reset()

                # -------- LIVE PLOT UPDATE throttle (~10s) --------
                now = time.time()
                if now - last_plot > 10:
                    plotter.update(
                        rewards_all_eps=rewards_log,
                        recent_means_all_eps=recent_mean_log,
                        lrs_all_eps=lr_log,
                        steps_all_eps=steps_log,
                        phase_idx=phase_idx,
                        save_path_png="car_racing/plots/live_training_status.png"
                    )
                    save_reward_curve(
                        rewards_log,
                        title="Reward curve (final phase PPO)",
                        save_path="car_racing/plots/reward_curve_final.png"
                    )
                    last_plot = now

        batch = {
            "obs":      torch.stack(buf_obs).to(device),
            "actions":  torch.stack(buf_actions).to(device),
            "logprobs": torch.stack(buf_logprobs).to(device),
            "rewards":  torch.stack(buf_rewards).to(device),
            "dones":    torch.stack(buf_dones).to(device),
            "values":   torch.stack(buf_values).to(device),
        }

        agent.update(batch)

    # END TRAIN
    torch.save(
        agent.model.state_dict(),
        "car_racing/checkpoints/ppo_carracing_cont_final_last.pth"
    )
    
    plotter.update(
        rewards_all_eps=rewards_log,
        recent_means_all_eps=recent_mean_log,
        lrs_all_eps=lr_log,
        steps_all_eps=steps_log,
        phase_idx=phase_idx,
        save_path_png="car_racing/plots/live_training_status.png"
    )

    save_reward_curve(
        rewards_log,
        title="Reward curve (final phase PPO)",
        save_path="car_racing/plots/reward_curve_final.png"
    )

    env.close()
    print("🏁 FINAL PHASE CONTINUOUS TRAIN DONE")


#################################################################
# 4. MAIN
#################################################################

if __name__ == "__main__":
    train_continuous_final()