"""
agent.py — Proximal Policy Optimisation (PPO) with GAE-λ advantages.

Architecture
------------
Shared MLP backbone (Tanh activations, orthogonal init) feeds two heads:

  Actor  — outputs Gaussian action distribution.
           mean  = tanh(linear)  ∈ (−1, 1)  (naturally bounded)
           std   = exp(learnable log_std parameter)

  Critic — scalar value estimate V(s).

The tanh squash on the actor mean keeps actions in [−1, 1] without an
explicit clamp during training, which simplifies the log-prob calculation
(we evaluate log_prob on the *unsquashed* sample, then clamp for env).

Key PPO details
---------------
  • GAE-λ advantage estimation (Schulman et al., 2016)
  • Clipped surrogate objective (ε = 0.2)
  • Entropy bonus to encourage exploration early on
  • Gradient clipping (max-norm = 0.5) for stability
  • Orthogonal weight initialisation (critical for PPO performance)
"""

from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ── Hyper-parameters ────────────────────────────────────────────────────
LR          = 3e-4
GAMMA       = 0.99
LAM         = 0.95    # GAE λ — higher → less bias, more variance
CLIP_EPS    = 0.2     # PPO clip ratio ε
VF_COEF     = 0.5     # value-loss coefficient
ENT_COEF    = 0.005   # entropy bonus coefficient (decay if over-exploring)
N_EPOCHS    = 10      # gradient update passes per rollout
BATCH_SIZE  = 64      # mini-batch size
HIDDEN_DIM  = 128


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 6, act_dim: int = 2):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM), nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(HIDDEN_DIM, act_dim)
        # Learnable log-std shared across states (common PPO practice)
        self.log_std    = nn.Parameter(torch.full((act_dim,), -0.5))
        self.critic     = nn.Linear(HIDDEN_DIM, 1)

        # Orthogonal initialisation — empirically important for PPO.
        # backbone: gain = √2 for Tanh
        # actor head: small gain so initial actions are near-zero
        # critic head: gain = 1
        for m in self.backbone:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor):
        h    = self.backbone(x)
        mean = torch.tanh(self.actor_mean(h))          # ∈ (−1, 1)
        std  = self.log_std.exp().clamp(1e-3, 1.0).expand_as(mean)
        val  = self.critic(h).squeeze(-1)
        return mean, std, val

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        """
        Sample an action from the policy.

        Returns
        -------
        action   : np.ndarray (act_dim,)  — clamped to [−1, 1]
        log_prob : float                   — log π(a|s)  (used by PPO)
        value    : float                   — V(s)        (used by GAE)
        """
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        mean, std, val = self(x)
        dist    = Normal(mean, std)
        raw     = dist.sample()                        # un-clamped sample
        action  = raw.clamp(-1.0, 1.0)
        log_p   = dist.log_prob(raw).sum(-1)           # sum over action dims
        return action.squeeze(0).numpy(), log_p.item(), val.item()


class PPO:
    """
    Wraps ActorCritic with a PPO update rule.

    Typical usage
    -------------
    agent = PPO()
    # ... collect rollout ...
    agent.update(rollout, last_value)
    """

    def __init__(self, obs_dim: int = 6, act_dim: int = 2):
        self.net = ActorCritic(obs_dim, act_dim)
        scheduler = optim.lr_scheduler.ExponentialLR(optim.Adam(self.net.parameters(), lr=LR), gamma=0.999)
        self.opt = optim.Adam(self.net.parameters(), lr=LR)

    # ================================================================== #
    #  PPO update
    # ================================================================== #

    def update(self, rollout: dict, last_value: float) -> dict:
        """
        Run N_EPOCHS of mini-batch PPO updates on a collected rollout.

        Parameters
        ----------
        rollout    : dict with keys obs, actions, log_probs, rewards,
                     values, dones  — each a list of length T.
        last_value : V(s_T)  — bootstrap value for GAE.

        Returns
        -------
        dict with mean policy_loss, value_loss, entropy over the update.
        """
        obs       = np.array(rollout["obs"],       dtype=np.float32)
        acts      = np.array(rollout["actions"],   dtype=np.float32)
        old_logps = np.array(rollout["log_probs"], dtype=np.float32)
        rewards   = np.array(rollout["rewards"],   dtype=np.float32)
        values    = np.array(rollout["values"],    dtype=np.float32)
        dones     = np.array(rollout["dones"],     dtype=np.float32)

        advantages, returns = self._gae(rewards, values, dones, last_value)

        # Normalise advantages within the rollout (stabilises learning)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t  = torch.tensor(obs)
        acts_t = torch.tensor(acts)
        olp_t  = torch.tensor(old_logps)
        adv_t  = torch.tensor(advantages)
        ret_t  = torch.tensor(returns)

        T = len(obs)
        metrics = {"policy": [], "value": [], "entropy": []}

        for _ in range(N_EPOCHS):
            for mb in self._minibatches(T):
                mean, std, vals = self.net(obs_t[mb])
                dist    = Normal(mean, std)
                log_p   = dist.log_prob(acts_t[mb]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                # Clipped surrogate objective
                ratio = (log_p - olp_t[mb]).exp()
                surr1 = ratio * adv_t[mb]
                surr2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t[mb]
                p_loss = -torch.min(surr1, surr2).mean()

                v_loss = (vals - ret_t[mb]).pow(2).mean()
                loss   = p_loss + VF_COEF * v_loss - ENT_COEF * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.opt.step()

                metrics["policy"].append(p_loss.item())
                metrics["value"].append(v_loss.item())
                metrics["entropy"].append(entropy.item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    # ================================================================== #
    #  GAE
    # ================================================================== #

    def _gae(
        self,
        rewards:    np.ndarray,
        values:     np.ndarray,
        dones:      np.ndarray,
        last_value: float,
    ):
        """
        Generalised Advantage Estimation (Schulman et al., 2016).

        δ_t = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
        A_t = δ_t + γλ · (1 − done_t) · A_{t+1}

        The per-step TD-error δ is the building block; GAE exponentially
        weights multi-step returns with decay γλ.  λ=1 → full Monte Carlo;
        λ=0 → TD(0).  λ=0.95 is a good trade-off.
        """
        T   = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            next_val  = last_value if t == T - 1 else values[t + 1]
            not_done  = 1.0 - float(dones[t])
            delta     = rewards[t] + GAMMA * next_val * not_done - values[t]
            gae       = delta + GAMMA * LAM * not_done * gae
            adv[t]    = gae

        returns = adv + values
        return adv, returns

    # ================================================================== #
    #  Helpers
    # ================================================================== #

    def _minibatches(self, T: int):
        idxs = np.random.permutation(T)
        for start in range(0, T, BATCH_SIZE):
            yield idxs[start: start + BATCH_SIZE]

    def save(self, path: str = "checkpoint.pt") -> None:
        torch.save(
            {"net": self.net.state_dict(), "opt": self.opt.state_dict()},
            path,
        )
        print(f"Checkpoint saved → {path}")

    def load(self, path: str = "checkpoint.pt") -> None:
        if not os.path.exists(path):
            print(f"No checkpoint found at '{path}' — starting fresh.")
            return
        ck = torch.load(path, map_location="cpu", weights_only=True)
        try:
            self.net.load_state_dict(ck["net"])
            self.opt.load_state_dict(ck["opt"])
            print(f"Checkpoint loaded ← {path}")
        except RuntimeError as err:
            print(
                "Checkpoint architecture mismatch; starting with fresh weights. "
                f"Reason: {err}"
            )
