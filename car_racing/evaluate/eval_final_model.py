import os
import time
import gymnasium as gym
import torch
import numpy as np

from car_racing.wrappers.grayscale_wrapper import GrayScaleWrapper
from car_racing.wrappers.frame_stack import FrameStack
from car_racing.models.cnn_policy_value_cont import CnnPolicyValueCont
from car_racing.agents.ppo_agent_cont import PPOAgentCont

# ================================================================
# make_env_eval()
# ------------------------------------------------
# Builds the evaluation environment:
# - Uses Gymnasium's CarRacing-v3 environment
# - Converts RGB frames to grayscale
# - Stacks 4 consecutive frames for temporal information
# ================================================================
def make_env_eval(render_mode="human", domain_randomize=False):
    base_env = gym.make(
        "CarRacing-v3",
        render_mode=render_mode,      # "human" to visualize, "rgb_array" for silent mode
        continuous=True,              # Continuous control mode
        domain_randomize=domain_randomize,  # Randomize track colors/textures (optional)
        lap_complete_percent=0.95,    # Episode ends when 95% of track is completed
    )

    # Convert observations to grayscale and stack 4 frames
    env = GrayScaleWrapper(base_env)
    env = FrameStack(env, k=4)

    return env

# ================================================================
# run_eval_episode()
# ------------------------------------------------
# Runs one full episode of evaluation using the trained PPO agent.
# It executes actions predicted by the model and collects rewards.
# ================================================================
@torch.no_grad()
def run_eval_episode(env, agent, device="cpu", max_steps=2000):
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        # Preprocess observation: (H,W,4) → (1,4,96,96) and normalize
        obs_t = torch.from_numpy(obs).float() / 255.0
        obs_t = obs_t.permute(2, 0, 1).unsqueeze(0).to(device)

        # Forward pass through the trained model
        mean, logstd, value = agent.model(obs_t)

        # Deterministic action: mean of Gaussian distribution
        act = mean.squeeze(0).cpu().numpy()

        # Clamp each action dimension to valid range for CarRacing
        # [steering, gas, brake]
        act[0] = np.clip(act[0], -1.0, 1.0)  # steering
        act[1] = np.clip(act[1],  0.0, 1.0)  # gas
        act[2] = np.clip(act[2],  0.0, 1.0)  # brake
        act = act.astype(np.float32)

        # Step in the environment with the chosen action
        next_obs, reward_env, terminated, truncated, env_info = env.step(act)
        done = terminated or truncated

        total_reward += reward_env
        steps += 1
        obs = next_obs

        # Stop if episode ends or max_steps reached
        if done or steps >= max_steps:
            break

    return total_reward, steps

# ================================================================
# eval_model()
# ------------------------------------------------
# Loads a trained PPO agent from a checkpoint and evaluates it
# over several test episodes. Displays the average, best, and
# worst scores achieved.
# ================================================================
def eval_model(
    checkpoint_path="car_racing/checkpoints/ppo_carracing_cont_final_best.pth",
    n_episodes=5,
    render_mode="human",
    device_str=None,
):

    # === Select device (GPU if available) ===
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print("Eval device:", device)

    # === Initialize environment ===
    env = make_env_eval(render_mode=render_mode, domain_randomize=False)

    # === Create model and PPO agent ===
    action_dim = env.action_space.shape[0]
    model = CnnPolicyValueCont(action_dim=action_dim)
    agent = PPOAgentCont(
        model=model,
        lr=5e-5,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.0,  # No exploration during evaluation
        update_epochs=1,
        minibatch_size=64,
        device=device,
    )

    # === Load pretrained weights ===
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    agent.model.load_state_dict(state_dict)
    agent.model.to(device)
    agent.model.eval()

    # === Run evaluation episodes ===
    scores = []
    steps_per_ep = []

    for ep in range(n_episodes):
        ep_reward, ep_steps = run_eval_episode(
            env=env,
            agent=agent,
            device=device,
            max_steps=2000,
        )
        scores.append(ep_reward)
        steps_per_ep.append(ep_steps)

        print(
            f"🏁 Eval ep {ep} done: "
            f"score={ep_reward:.2f}  steps={ep_steps}"
        )

        # Short pause between episodes for rendering clarity
        if render_mode == "human":
            time.sleep(0.5)

    # Close environment
    env.close()

    # === Compute statistics ===
    mean_score = sum(scores) / len(scores)
    best_score = max(scores)
    worst_score = min(scores)
    avg_len = sum(steps_per_ep) / len(steps_per_ep)

    # === Display results ===
    print("=====================================")
    print(f"✅ Evaluation finished on {n_episodes} episodes")
    print(f"   Average score: {mean_score:.2f}")
    print(f"   Best:          {best_score:.2f}")
    print(f"   Worst:         {worst_score:.2f}")
    print(f"   Avg length:    {avg_len:.1f} steps")
    print("=====================================")


# ================================================================
# MAIN EXECUTION
# ------------------------------------------------
# This section allows the file to be run directly.
# It loads the best model and evaluates it over 5 episodes.
# ================================================================
if __name__ == "__main__":
    eval_model(
        checkpoint_path="car_racing/checkpoints/ppo_carracing_cont_final_best.pth",
        n_episodes=5,
        render_mode="human",
    )
