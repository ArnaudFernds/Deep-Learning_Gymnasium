import torch
import torch.nn as nn
from torch.distributions import Normal


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
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.device = device

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):

        # (H,W,4) uint8 -> (1,4,96,96) float32 [0,1]
        obs_t = torch.from_numpy(obs).float() / 255.0
        obs_t = obs_t.permute(2, 0, 1).unsqueeze(0).to(self.device)

        mean, logstd, value = self.model(obs_t)  # mean:(1,3), logstd:(1,3), value:(1,1)
        std = torch.exp(logstd)
        dist = Normal(mean, std)

        if deterministic:
            act_t = mean
        else:
            act_t = dist.sample()

        logprob = dist.log_prob(act_t).sum(dim=-1)
        value = value.squeeze(-1)

        action_np = act_t.squeeze(0).cpu().numpy()

        return action_np, logprob.item(), value.item()

    def compute_gae(self, rewards, values, dones, last_value):
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0

        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t+1]
            delta = (
                rewards[t]
                + self.gamma * next_value * (1 - dones[t])
                - values[t]
            )
            gae = (
                delta
                + self.gamma * self.lam * (1 - dones[t]) * gae
            )
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, buffer):
        obs_all      = buffer["obs"].to(self.device)
        acts_all     = buffer["actions"].to(self.device)
        old_logp_all = buffer["logprobs"].to(self.device)
        vals_all     = buffer["values"].to(self.device)
        rews_all     = buffer["rewards"].to(self.device)
        dones_all    = buffer["dones"].to(self.device)

        # bootstrap last state value
        with torch.no_grad():
            last_obs = obs_all[-1].unsqueeze(0)
            mean, logstd, last_val = self.model(last_obs)
            last_val = last_val.squeeze(-1)

        adv, ret = self.compute_gae(rews_all, vals_all, dones_all, last_val)

        # adv normalize
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        T = obs_all.shape[0]

        for _ in range(self.update_epochs):
            perm = torch.randperm(T, device=self.device)
            for start in range(0, T, self.minibatch_size):
                mb_idx = perm[start:start + self.minibatch_size]

                mb_obs   = obs_all[mb_idx]
                mb_acts  = acts_all[mb_idx]
                mb_oldlp = old_logp_all[mb_idx]
                mb_adv   = adv[mb_idx]
                mb_ret   = ret[mb_idx]

                mean, logstd, value_pred = self.model(mb_obs)
                std = torch.exp(logstd)
                dist = Normal(mean, std)

                new_logp = dist.log_prob(mb_acts).sum(dim=-1)

                ratio = torch.exp(new_logp - mb_oldlp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = value_pred.squeeze(-1)  # (mb,)
                value_loss = 0.5 * (mb_ret - value_pred).pow(2).mean()

                entropy = dist.entropy().sum(dim=-1).mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()