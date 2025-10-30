import torch
import torch.nn as nn
from torch.distributions import Normal

# ================================================================
# PPOAgentCont : Proximal Policy Optimization Agent (Continuous)
# ================================================================
# This agent learns a continuous control policy to drive a car in
# the CarRacing-v3 environment. It uses PPO, a stable policy-
# gradient algorithm that updates the policy carefully by clipping
# the probability ratio between new and old actions.
# ================================================================

class PPOAgentCont:
    def __init__(self,
                 model,
                 lr=3e-4,
                 gamma=0.99,
                 lam=0.95,
                 clip_eps=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 update_epochs=4,
                 minibatch_size=64,
                 device="cpu"):

        # The neural network model (CNN + 2 heads: policy and value)
        self.model = model.to(device)

        # Optimizer (Adam)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # PPO hyperparameters
        self.gamma = gamma                # Discount factor (future reward importance)
        self.lam = lam                    # GAE lambda (bias/variance trade-off)
        self.clip_eps = clip_eps          # Clipping range for PPO objective
        self.value_coef = value_coef      # Weight of value loss
        self.entropy_coef = entropy_coef  # Weight of entropy bonus (exploration)
        self.update_epochs = update_epochs  # How many passes over the same rollout
        self.minibatch_size = minibatch_size # Mini-batch size for PPO updates
        self.device = device

    # =============================================================
    # SELECT_ACTION : choose an action given the current observation
    # =============================================================
    @torch.no_grad()
    def select_action(self, obs, deterministic=False):

        # Convert observation from (H,W,4) uint8 → (1,4,96,96) float32 [0,1]
        obs_t = torch.from_numpy(obs).float() / 255.0
        obs_t = obs_t.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Forward pass through the model
        # Outputs: mean (1,3), logstd (1,3), value (1,1)
        mean, logstd, value = self.model(obs_t)
        std = torch.exp(logstd)
        dist = Normal(mean, std)

        # Deterministic (for evaluation) or sampled (for exploration) action
        if deterministic:
            act_t = mean
        else:
            act_t = dist.sample()

        # Log probability of the chosen action (for PPO loss)
        logprob = dist.log_prob(act_t).sum(dim=-1)

        # Value prediction (V(s)) for the critic
        value = value.squeeze(-1)

        # Convert to numpy for the environment
        action_np = act_t.squeeze(0).cpu().numpy()

        # Return the action, log-probability, and estimated value
        return action_np, logprob.item(), value.item()

    # =============================================================
    # COMPUTE_GAE : Generalized Advantage Estimation
    # =============================================================
    def compute_gae(self, rewards, values, dones, last_value):
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0

        # Iterate backward through the rollout to compute advantages
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]

            # Temporal difference error (TD error)
            delta = (
                rewards[t]
                + self.gamma * next_value * (1 - dones[t])
                - values[t]
            )

            # Recursive GAE formula
            gae = (
                delta
                + self.gamma * self.lam * (1 - dones[t]) * gae
            )
            advantages[t] = gae

        # Returns = advantages + predicted values
        returns = advantages + values
        return advantages, returns

    # =============================================================
    # UPDATE : Train the model using the PPO clipped objective
    # =============================================================
    def update(self, buffer):
        # Load rollout data from the buffer
        obs_all      = buffer["obs"].to(self.device)
        acts_all     = buffer["actions"].to(self.device)
        old_logp_all = buffer["logprobs"].to(self.device)
        vals_all     = buffer["values"].to(self.device)
        rews_all     = buffer["rewards"].to(self.device)
        dones_all    = buffer["dones"].to(self.device)

        # Bootstrap last state value (if episode not terminated)
        with torch.no_grad():
            last_obs = obs_all[-1].unsqueeze(0)
            mean, logstd, last_val = self.model(last_obs)
            last_val = last_val.squeeze(-1)

        # Compute advantages and returns
        adv, ret = self.compute_gae(rews_all, vals_all, dones_all, last_val)

        # Normalize advantages for stability
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        T = obs_all.shape[0]

        # =========================================================
        # PPO optimization loop
        # =========================================================
        for _ in range(self.update_epochs):

            # Shuffle indices for mini-batch sampling
            perm = torch.randperm(T, device=self.device)

            # Iterate over mini-batches
            for start in range(0, T, self.minibatch_size):
                mb_idx = perm[start:start + self.minibatch_size]

                # Extract mini-batch samples
                mb_obs   = obs_all[mb_idx]
                mb_acts  = acts_all[mb_idx]
                mb_oldlp = old_logp_all[mb_idx]
                mb_adv   = adv[mb_idx]
                mb_ret   = ret[mb_idx]

                # Forward pass through the model
                mean, logstd, value_pred = self.model(mb_obs)
                std = torch.exp(logstd)
                dist = Normal(mean, std)

                # Compute new log-probabilities
                new_logp = dist.log_prob(mb_acts).sum(dim=-1)

                # PPO ratio between new and old policy
                ratio = torch.exp(new_logp - mb_oldlp)

                # Clipped objective
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (critic)
                value_pred = value_pred.squeeze(-1)
                value_loss = 0.5 * (mb_ret - value_pred).pow(2).mean()

                # Entropy (encourages exploration)
                entropy = dist.entropy().sum(dim=-1).mean()

                # Total PPO loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (for numerical stability)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                # Optimization step
                self.optimizer.step()
