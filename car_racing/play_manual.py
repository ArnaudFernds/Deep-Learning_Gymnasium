import gymnasium as gym
import numpy as np
import pygame


def main():
    # 1. Crée l'env en continuous EXACTEMENT comme ton training "dr=False"
    env = gym.make(
        "CarRacing-v3",
        render_mode="human",         # ouvre une fenêtre interactive
        continuous=True,             # on contrôle [steer, gas, brake]
        domain_randomize=False,      # piste pas randomisée (stable)
        lap_complete_percent=0.95,
    )

    print("\n🚗 Contrôles du jeu CarRacing-v3 (mode manuel) :")
    print("  ↑ : accélérer")
    print("  ↓ : freiner / marche arrière")
    print("  ← : tourner à gauche")
    print("  → : tourner à droite")
    print("  R : reset l'épisode")
    print("  ESC : quitter\n")

    obs, info = env.reset()
    total_reward = 0.0
    running = True

    clock = pygame.time.Clock()

    while running:
        # 2. On lit les events pour que la fenêtre reste responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # 3. Action continue [steer, gas, brake]
        steer = 0.0
        gas = 0.0
        brake = 0.0

        if keys[pygame.K_LEFT]:
            steer = -1.0
        if keys[pygame.K_RIGHT]:
            steer = +1.0
        if keys[pygame.K_UP]:
            gas = 1.0
        if keys[pygame.K_DOWN]:
            brake = 1.0  # frein à fond (0..1)

        if keys[pygame.K_r]:
            print("🔄 reset manual")
            obs, info = env.reset()
            total_reward = 0.0
            continue

        if keys[pygame.K_ESCAPE]:
            print("👋 Bye!")
            break

        # 4. Convertir en np.array float32 pour Gym
        action = np.array([steer, gas, brake], dtype=np.float32)

        # 5. Step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # 6. Fin d'épisode ?
        if terminated or truncated:
            print(f"🏁 Épisode terminé, score = {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0

        # 7. Petit throttle pour pas bouffer 1000 FPS
        clock.tick(60)  # ~60Hz

    env.close()


if __name__ == "__main__":
    main()