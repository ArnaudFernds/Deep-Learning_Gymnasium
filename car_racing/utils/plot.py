import os
import matplotlib.pyplot as plt


def save_reward_curve(rewards, title, save_path, xlabel="Episode", ylabel="Total Reward"):
    if len(rewards) == 0:
        return

    plt.figure()
    plt.plot(rewards)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()