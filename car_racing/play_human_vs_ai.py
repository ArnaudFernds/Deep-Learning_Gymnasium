import pygame
import numpy as np
import torch
import gymnasium as gym
import time

from car_racing.wrappers.grayscale_wrapper import GrayScaleWrapper
from car_racing.wrappers.frame_stack import FrameStack
from car_racing.models.cnn_policy_value_cont import CnnPolicyValueCont


########################################################
# Environments
########################################################

def make_env_continuous_play(domain_randomize=False):
    base_env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        continuous=True,
        domain_randomize=domain_randomize,
        lap_complete_percent=0.95,
    )
    return base_env


def make_env_agent(domain_randomize=False):
    base_env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        continuous=True,
        domain_randomize=domain_randomize,
        lap_complete_percent=0.95,
    )
    env = GrayScaleWrapper(base_env)
    env = FrameStack(env, k=4)
    return env


########################################################
# Helpers
########################################################

def obs_to_torch(obs_np: np.ndarray, device: torch.device):
    obs_t = torch.from_numpy(obs_np).float() / 255.0
    obs_t = obs_t.permute(2, 0, 1).unsqueeze(0).to(device)
    return obs_t


@torch.no_grad()
def agent_select_action(model, obs_np, device):
    x = obs_to_torch(obs_np, device)
    mu, _logstd, _value = model.policy_forward(x)
    action = mu.squeeze(0).cpu().numpy()

    action[0] = np.clip(action[0], -1.0, 1.0)  # steer
    action[1] = np.clip(action[1],  0.0, 1.0)  # gas
    action[2] = np.clip(action[2],  0.0, 1.0)  # brake

    return action.astype(np.float32)


def get_human_action(keys):
    steer = 0.0
    gas = 0.0
    brake = 0.0

    if keys[pygame.K_LEFT]:
        steer -= 1.0
    if keys[pygame.K_RIGHT]:
        steer += 1.0
    if keys[pygame.K_UP]:
        gas = 1.0
    if keys[pygame.K_DOWN]:
        brake = 1.0
        
    steer = np.clip(steer, -1.0, 1.0)
    gas   = np.clip(gas,   0.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)

    return np.array([steer, gas, brake], dtype=np.float32)


########################################################
# Main loop
########################################################

def play_human_vs_agent(
    model_path="car_racing/checkpoints/ppo_carracing_cont_final_best.pth",
    fps=50,
    domain_randomize=False,
    device_str=None
):

    # ===== Device =====
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print("Device:", device)

    # ===== Model load =====
    dummy_env_agent = make_env_agent(domain_randomize=domain_randomize)
    action_dim = dummy_env_agent.action_space.shape[0]

    model = CnnPolicyValueCont(action_dim=action_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Loaded agent weights from {model_path}")

    # ===== Real envs =====
    env_human = make_env_continuous_play(domain_randomize=domain_randomize)
    env_agent = dummy_env_agent

    obs_human, info_h = env_human.reset()
    obs_agent, info_a = env_agent.reset()

    ep_reward_human = 0.0
    ep_reward_agent = 0.0

    # ===== pygame setup =====
    pygame.init()

    SCALE = 4
    cam_w = 96 * SCALE
    cam_h = 96 * SCALE

    window_w = cam_w
    window_h = cam_h * 2

    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption("CarRacing-v3: YOU (top) vs AGENT (bottom)")

    font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    running = True
    while running:
        # ---- events / quit ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # ---- 1. HUMAN STEP ----
        keys = pygame.key.get_pressed()
        human_action = get_human_action(keys)

        next_obs_h, rew_h, term_h, trunc_h, info_h = env_human.step(human_action)
        ep_reward_human += rew_h

        if term_h or trunc_h:
            print(f"👤 Human episode finished. Reward={ep_reward_human:.2f}")
            obs_human, info_h = env_human.reset()
            ep_reward_human = 0.0
        else:
            obs_human = next_obs_h

        # ---- 2. AGENT STEP ----
        agent_action = agent_select_action(model, obs_agent, device)

        next_obs_a, rew_a, term_a, trunc_a, info_a = env_agent.step(agent_action)
        ep_reward_agent += rew_a

        if term_a or trunc_a:
            print(f"🤖 Agent episode finished. Reward={ep_reward_agent:.2f}")
            obs_agent, info_a = env_agent.reset()
            ep_reward_agent = 0.0
        else:
            obs_agent = next_obs_a

        # ---- 3. RENDER BOTH ----
        frame_h = env_human.render()
        frame_a = env_agent.render()

        surf_h = pygame.surfarray.make_surface(np.transpose(frame_h, (1, 0, 2)))
        surf_a = pygame.surfarray.make_surface(np.transpose(frame_a, (1, 0, 2)))

        surf_h = pygame.transform.scale(surf_h, (cam_w, cam_h))
        surf_a = pygame.transform.scale(surf_a, (cam_w, cam_h))

        screen.blit(surf_h, (0, 0))
        screen.blit(surf_a, (0, cam_h))

        # ---- 4. HUD text ----
        txt_h = f"YOU   reward:{ep_reward_human:.1f}"
        txt_a = f"AGENT reward:{ep_reward_agent:.1f}"
        txt_fps = f"FPS:{clock.get_fps():.1f}"

        color_white = (255, 255, 255)
        outline_black = (0, 0, 0)

        def draw_text_with_outline(msg, x, y):
            img_main = font.render(msg, True, color_white)
            img_shadow = font.render(msg, True, outline_black)

            screen.blit(img_shadow, (x + 1, y + 1))
            screen.blit(img_shadow, (x - 1, y - 1))
            screen.blit(img_shadow, (x + 1, y - 1))
            screen.blit(img_shadow, (x - 1, y + 1))

            screen.blit(img_main, (x, y))

        draw_text_with_outline(txt_h, 10, 10)
        draw_text_with_outline(txt_a, 10, cam_h + 10)
        draw_text_with_outline(txt_fps, window_w - 120, 10)

        pygame.display.flip()

        # ---- 5. Limit FPS ----
        clock.tick(fps)

    pygame.quit()
    env_human.close()
    env_agent.close()
    print("Bye 👋")


if __name__ == "__main__":
    play_human_vs_agent()