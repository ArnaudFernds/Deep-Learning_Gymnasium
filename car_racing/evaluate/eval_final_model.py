import os
import time
import gymnasium as gym
import torch
import numpy as np

from car_racing.wrappers.grayscale_wrapper import GrayScaleWrapper
from car_racing.wrappers.frame_stack import FrameStack
from car_racing.models.cnn_policy_value_cont import CnnPolicyValueCont
from car_racing.agents.ppo_agent_cont import PPOAgentCont


def make_env_eval(render_mode="human", domain_randomize=False):
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


@torch.no_grad()
def run_eval_episode(env, agent, device="cpu", max_steps=2000):
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        obs_t = torch.from_numpy(obs).float() / 255.0
        obs_t = obs_t.permute(2, 0, 1).unsqueeze(0).to(device)

        mean, logstd, value = agent.model(obs_t)
        act = mean.squeeze(0).cpu().numpy()

        # clamp CarRacing: [steer, gas, brake]
        act[0] = np.clip(act[0], -1.0, 1.0)
        act[1] = np.clip(act[1],  0.0, 1.0)
        act[2] = np.clip(act[2],  0.0, 1.0)
        act = act.astype(np.float32)

        next_obs, reward_env, terminated, truncated, env_info = env.step(act)
        done = terminated or truncated

        total_reward += reward_env
        steps += 1
        obs = next_obs

        if done or steps >= max_steps:
            break

    return total_reward, steps


def eval_model(
    checkpoint_path="car_racing/checkpoints/ppo_carracing_cont_final_best.pth",
    n_episodes=5,
    render_mode="human",
    device_str=None,
):

    # === device ===
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print("Eval device:", device)

    # === env ===
    env = make_env_eval(render_mode=render_mode, domain_randomize=False)

    # === model & agent ===
    action_dim = env.action_space.shape[0]
    model = CnnPolicyValueCont(action_dim=action_dim)
    agent = PPOAgentCont(
        model=model,
        lr=5e-5,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        update_epochs=1,
        minibatch_size=64,
        device=device,
    )

    # load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint introuvable: {checkpoint_path}"
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    agent.model.load_state_dict(state_dict)
    agent.model.to(device)
    agent.model.eval()

    # === run episodes ===
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

        if render_mode == "human":
            time.sleep(0.5)

    env.close()

    mean_score = sum(scores) / len(scores)
    best_score = max(scores)
    worst_score = min(scores)
    avg_len = sum(steps_per_ep) / len(steps_per_ep)

    print("=====================================")
    print(f"✅ Evaluation terminée sur {n_episodes} épisodes")
    print(f"   Moyenne score: {mean_score:.2f}")
    print(f"   Meilleur:      {best_score:.2f}")
    print(f"   Pire:          {worst_score:.2f}")
    print(f"   Longueur moy:  {avg_len:.1f} steps")
    print("=====================================")


if __name__ == "__main__":
    eval_model(
        checkpoint_path="car_racing/checkpoints/ppo_carracing_cont_final_best.pth",
        n_episodes=5,
        render_mode="human",
    )