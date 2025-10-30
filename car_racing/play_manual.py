import gymnasium as gym
import numpy as np
import pygame

# ================================================================
# Manual CarRacing-v3 Player
# ------------------------------------------------
# This script allows a human player to manually control the car
# in the CarRacing-v3 environment using keyboard inputs.
# Controls are mapped through Pygame.
# ================================================================

def main():
    # 1. Create the environment (same setup as in training)
    env = gym.make(
        "CarRacing-v3",
        render_mode="human",         # Opens an interactive display window
        continuous=True,             # Continuous control: [steer, gas, brake]
        domain_randomize=False,      # Fixed visuals (no random track colors)
        lap_complete_percent=0.95,   # End episode at 95% lap completion
    )

    # 2. Display control instructions
    print("\n🚗 CarRacing-v3 Controls (Manual Mode):")
    print("  ↑ : accelerate")
    print("  ↓ : brake / reverse")
    print("  ← : steer left")
    print("  → : steer right")
    print("  R : reset episode")
    print("  ESC : quit\n")

    # 3. Initialize environment and variables
    obs, info = env.reset()
    total_reward = 0.0
    running = True

    clock = pygame.time.Clock()  # Controls the frame rate (≈60 FPS)

    # =============================================================
    # MAIN GAME LOOP
    # =============================================================
    while running:
        # Process window events (so the window stays responsive)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Check which keys are currently pressed
        keys = pygame.key.get_pressed()

        # Initialize continuous control inputs
        steer = 0.0
        gas = 0.0
        brake = 0.0

        # Update action values based on user input
        if keys[pygame.K_LEFT]:
            steer = -1.0
        if keys[pygame.K_RIGHT]:
            steer = +1.0
        if keys[pygame.K_UP]:
            gas = 1.0
        if keys[pygame.K_DOWN]:
            brake = 1.0  # full brake (range 0..1)

        # Reset environment manually
        if keys[pygame.K_r]:
            print("🔄 Manual reset")
            obs, info = env.reset()
            total_reward = 0.0
            continue

        # Quit game
        if keys[pygame.K_ESCAPE]:
            print("👋 Bye!")
            break

        # 4. Build continuous action array [steer, gas, brake]
        action = np.array([steer, gas, brake], dtype=np.float32)

        # 5. Step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # 6. End of episode?
        if terminated or truncated:
            print(f"🏁 Episode finished, score = {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0

        # 7. Limit to 60 frames per second (avoids CPU overload)
        clock.tick(60)

    # 8. Cleanup
    env.close()


# =============================================================
# Entry point
# =============================================================
if __name__ == "__main__":
    main()
