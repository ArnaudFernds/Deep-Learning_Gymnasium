# 🏎️ Reinforcement Learning – CarRacing-v3 (PPO Continuous)

This project implements a **Proximal Policy Optimization (PPO)** agent trained on the **CarRacing-v3** environment from Gymnasium.  
The goal: teach a virtual driver to master randomly generated racing tracks using only **raw pixel observations** — no physics, no rules, just reinforcement learning.

---

## 📘 Overview

- **Environment:** [CarRacing-v3 – Gymnasium](https://gymnasium.farama.org/environments/box2d/car_racing/)
- **Algorithm:** Proximal Policy Optimization (PPO)
- **Framework:** PyTorch
- **Mode:** Continuous control (steering, gas, brake)
- **Training steps:** 6,000,000+
- **Average performance:** ~900 reward (stable lap completion)

The project includes:
- A **custom training pipeline** with a 4-phase learning rate & entropy scheduler  
- Real-time logging and reward plotting  
- A **“Human vs AI”** race mode using Pygame  
- A detailed research **report (PDF)** explaining the methodology, experiments, and results  

---

## 🧠 Architecture

The agent consists of:
- A **CNN encoder** processing 4 stacked grayscale frames (96×96×4)
- Two output heads:
  - 🎮 **Actor:** outputs a Gaussian policy (mean & std) → continuous steering, gas, and brake  
  - 💰 **Critic:** estimates the state value for advantage calculation
- Optimization with **Adam** and **GAE-Lambda** for stable advantage estimation.

Training phases progressively reduce exploration (entropy) and learning rate to refine the driving behavior.

---

## ⚙️ Training Setup

| Phase | Steps      | Learning Rate | Entropy Coef | Rollout Horizon |
|-------|-------------|----------------|---------------|-----------------|
| 1 | 0 – 1.5M | 5e-5 | 1e-3 | 4096 |
| 2 | 1.5M – 3M | 3e-5 | 7e-4 | 3072 |
| 3 | 3M – 4.5M | 1.5e-5 | 4e-4 | 2048 |
| 4 | 4.5M – 6M | 5e-6 | 1e-4 | 1024 |

### Key Hyperparameters
- **Discount factor (γ):** 0.99  
- **GAE lambda (λ):** 0.95  
- **Clip ε:** 0.2  
- **Batch size:** 64  
- **Value loss coef:** 0.5  
- **Entropy coef:** scheduled decay  

---
