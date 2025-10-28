import matplotlib.pyplot as plt
import numpy as np

class LivePlotter:
    def __init__(self):
        plt.ion()
        self.fig, self.ax_left = plt.subplots()
        self.ax_right = self.ax_left.twinx()

        self.fig.suptitle("CarRacing PPO Training (live)", fontsize=14)

        self.line_rewards, = self.ax_left.plot([], [], label="Episode reward", alpha=0.3)
        self.line_mean,    = self.ax_left.plot([], [], label="Recent mean (top15/20)", linewidth=2)

        self.line_lr,      = self.ax_right.plot([], [], label="LR", linestyle="--", linewidth=1)

        self.ax_left.set_xlabel("Episode")
        self.ax_left.set_ylabel("Reward")
        self.ax_right.set_ylabel("LR")

        self.ax_left.legend(loc="upper left")
        self.ax_right.legend(loc="upper right")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(
        self,
        rewards_all_eps,
        recent_means_all_eps,
        lrs_all_eps,
        steps_all_eps,
        phase_idx,
        save_path_png=None
    ):

        if len(rewards_all_eps) == 0:
            return

        x = np.arange(len(rewards_all_eps))

        # gauche
        self.line_rewards.set_data(x, rewards_all_eps)
        self.line_mean.set_data(x, recent_means_all_eps)

        self.ax_left.relim()
        self.ax_left.autoscale_view()

        # droite
        self.line_lr.set_data(x, lrs_all_eps)
        self.ax_right.relim()
        self.ax_right.autoscale_view()

        # titre avec infos
        last_step = steps_all_eps[-1]
        last_mean = recent_means_all_eps[-1]
        last_lr   = lrs_all_eps[-1]
        self.fig.suptitle(
            f"CarRacing PPO Live | step={last_step} | phase={phase_idx} | "
            f"recent_mean={last_mean:.1f} | lr={last_lr:.2e}",
            fontsize=14
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if save_path_png is not None:
            self.fig.savefig(save_path_png, dpi=150)