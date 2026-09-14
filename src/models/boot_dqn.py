"""Bootstrapped DQN with randomized prior functions, as a Stable-Baselines3 model.

Osband et al. 2016 (Bootstrapped DQN) + Osband, Aslanides & Cassirer 2018
(Randomized Prior Functions). Mirrors the bsuite `boot_dqn` reference agent:

  * K independent MLP Q-networks (no shared torso), run as one batched network.
  * Q_k(s, a) = f_k(s, a) + prior_scale * p_k(s, a), with p_k a frozen random MLP.
  * Each transition gets a bootstrap mask m ~ Bernoulli(mask_prob)^K; head k only
    trains on transitions with m_k = 1.
  * One head is sampled at the start of every episode and followed greedily for
    the whole episode (no epsilon-greedy).
  * predict(deterministic=True), used for greedy evaluation, acts on the mean Q
    over heads.

Drop-in for the SB3 DQN API: BootDQN(policy="MlpPolicy", env=env, ...).
Only supports a single (non-vectorised) environment.
"""

import math

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, preprocess_obs
from torch import nn


# ----------------------------------------------------------------- networks
class EnsembleLinear(nn.Module):
    """K independent linear layers applied in one batched matmul: (K,B,i) -> (K,B,o)."""

    def __init__(self, n_heads, in_dim, out_dim):
        super().__init__()
        bound = 1.0 / math.sqrt(in_dim)  # same scale as nn.Linear's default init
        self.weight = nn.Parameter(th.empty(n_heads, in_dim, out_dim).uniform_(-bound, bound))
        self.bias = nn.Parameter(th.empty(n_heads, 1, out_dim).uniform_(-bound, bound))

    def forward(self, x):
        return th.baddbmm(self.bias, x, self.weight)


class EnsembleMLP(nn.Module):
    def __init__(self, n_heads, in_dim, out_dim, net_arch):
        super().__init__()
        layers, last = [], in_dim
        for h in net_arch:
            layers += [EnsembleLinear(n_heads, last, h), nn.ReLU()]
            last = h
        layers.append(EnsembleLinear(n_heads, last, out_dim))
        self.net = nn.Sequential(*layers)
        self.n_heads = n_heads

    def forward(self, x):  # x: (B, in) -> (K, B, out)
        return self.net(x.unsqueeze(0).expand(self.n_heads, -1, -1))


class EnsembleQNetwork(nn.Module):
    """Trainable ensemble + frozen random prior. forward(obs) -> Q of shape (K, B, A)."""

    def __init__(self, observation_space, action_space, net_arch, n_heads, prior_scale, normalize_images=True):
        super().__init__()
        self.observation_space = observation_space
        self.normalize_images = normalize_images
        in_dim = get_flattened_obs_dim(observation_space)
        n_actions = int(action_space.n)
        self.prior_scale = prior_scale
        self.trainable = EnsembleMLP(n_heads, in_dim, n_actions, net_arch)
        self.prior = EnsembleMLP(n_heads, in_dim, n_actions, net_arch)
        # Frozen, but kept as parameters so the target network copies the same prior.
        self.prior.requires_grad_(False)

    def forward(self, obs):
        x = preprocess_obs(obs, self.observation_space, self.normalize_images).float().flatten(1)
        return self.trainable(x) + self.prior_scale * self.prior(x)


# ------------------------------------------------------------------- policy
class BootDQNPolicy(BasePolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        n_heads=20,
        prior_scale=3.0,
        normalize_images=True,
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=None,
        **_ignored,  # e.g. features_extractor_class, activation_fn from shared kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        assert isinstance(action_space, spaces.Discrete), "BootDQN needs a discrete action space"
        self.net_arch = list(net_arch) if net_arch is not None else [64, 64]
        self.n_heads = n_heads
        self.prior_scale = prior_scale
        self.active_head = 0

        make = lambda: EnsembleQNetwork(
            observation_space, action_space, self.net_arch, n_heads, prior_scale, normalize_images
        )
        self.q_net = make()
        self.q_net_target = make()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.train(False)

        self.optimizer = self.optimizer_class(
            [p for p in self.q_net.parameters() if p.requires_grad],
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        self.resample_head()

    def resample_head(self):
        self.active_head = int(np.random.randint(self.n_heads))

    def forward(self, obs, deterministic=True):
        return self._predict(obs, deterministic)

    def _predict(self, obs, deterministic=True):
        q = self.q_net(obs)  # (K, B, A)
        if deterministic:  # evaluation: act on the ensemble mean
            return q.mean(dim=0).argmax(dim=1)
        return q[self.active_head].argmax(dim=1)  # behaviour: the sampled head

    def set_training_mode(self, mode):
        self.q_net.train(mode)
        self.training = mode

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(net_arch=self.net_arch, n_heads=self.n_heads, prior_scale=self.prior_scale)
        return data


# ------------------------------------------------------------ replay buffer
class MaskedReplayBuffer(ReplayBuffer):
    """ReplayBuffer that also stores a Bernoulli bootstrap mask per transition.

    After each sample(), the masks of the sampled batch are in `self.last_masks` (B, K).
    """

    def __init__(self, *args, n_heads=20, mask_prob=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.n_envs == 1, "MaskedReplayBuffer supports a single env only"
        self.n_heads, self.mask_prob = n_heads, mask_prob
        self.masks = np.ones((self.buffer_size, n_heads), dtype=np.float32)
        self.last_masks = None

    def add(self, *args, **kwargs):
        self.masks[self.pos] = np.random.rand(self.n_heads) < self.mask_prob
        super().add(*args, **kwargs)

    def _get_samples(self, batch_inds, env=None):
        self.last_masks = self.to_torch(self.masks[batch_inds])
        return super()._get_samples(batch_inds, env)


# -------------------------------------------------------------------- agent
class BootDQN(DQN):
    def __init__(self, policy="MlpPolicy", env=None, n_heads=20, prior_scale=3.0, mask_prob=1.0, **kwargs):
        # `policy` is accepted for API compatibility with DQN; BootDQNPolicy is always used.
        policy_kwargs = dict(kwargs.pop("policy_kwargs", None) or {})
        policy_kwargs.update(n_heads=n_heads, prior_scale=prior_scale)
        # Exploration comes from the sampled head, not from epsilon.
        kwargs.pop("exploration_initial_eps", None)
        kwargs.pop("exploration_final_eps", None)
        
        # clipping is effectively off by default (bsuite does not clip).
        kwargs.setdefault("max_grad_norm", 1e6)
        super().__init__(
            BootDQNPolicy,
            env,
            policy_kwargs=policy_kwargs,
            replay_buffer_class=MaskedReplayBuffer,
            replay_buffer_kwargs=dict(n_heads=n_heads, mask_prob=mask_prob),
            exploration_initial_eps=0.0,
            exploration_final_eps=0.0,
            **kwargs,
        )
        assert self.n_envs == 1, "BootDQN supports a single env only"

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
        if dones[0]:  # new episode -> new Thompson sample
            self.policy.resample_head()

    def train(self, gradient_steps, batch_size=100):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            masks = self.replay_buffer.last_masks.T  # (K, B)
            rewards = data.rewards.view(1, -1)
            not_done = 1.0 - data.dones.view(1, -1)

            with th.no_grad():
                # Each head bootstraps from its own target head.
                next_q = self.q_net_target(data.next_observations).max(dim=2).values  # (K, B)
                target = rewards + not_done * self.gamma * next_q

            q = self.q_net(data.observations)  # (K, B, A)
            idx = data.actions.long().view(1, -1, 1).expand(q.shape[0], -1, 1)
            q_a = q.gather(2, idx).squeeze(2)  # (K, B)

            # Masked squared TD error: mean over each head's own samples, summed over
            # heads so every head gets the gradient scale of a standalone DQN.
            sq = 0.5 * (q_a - target) ** 2
            loss = ((sq * masks).sum(dim=1) / masks.sum(dim=1).clamp(min=1.0)).sum()
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.q_net.trainable.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
