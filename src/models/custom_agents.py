from __future__ import annotations

import copy


import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.utils import (
    ConstantSchedule,
    polyak_update,
)
from torch.nn import functional as F

from utils.state_bank import StateBank

#############################################################
## Helper

# Sample from the convex hull of k embeddings of real inputs 
# and use dirichlet as coefficients for sampling
def sample_hull(P, n=1000, alpha=1.0, rng=np.random.default_rng()):
    # P: (k, d) vertices. Returns (n, d) points in conv(P).
    w = rng.dirichlet(np.full(len(P), alpha), size=n)  # (n, k)
    return th.tensor(w, dtype=th.float32) @ P


#############################################################

class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # --- the only change vs. SB3's DQN ---
                # action selection with the ONLINE net
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                # action evaluation with the TARGET net
                next_q_values = th.gather(
                    self.q_net_target(replay_data.next_observations), dim=1, index=next_actions
                )
                # --------------------------------------
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


#############################################################

r"""
    Baseline Algorithms extended by SsVD (DQN_F, SAC_F). Next to the additional hyper parameters...

    eta_forgetting ($b$)
    baseline ($b$)
    forget_ratio ($\xi$)

    ...each model receives a "world_model", which deteermines how samples are generated for 
    the SsVD updates this can be one of either options:

    DQN_F:      [conv_hull_img, env, generated, stored_data, None, Object]

    SAC_F:      [env, generated, None, Object]

    "env": use the stored observation space's "sample()" method

    "stored_data": refered to pre generated data using the "StateBank" class

    "generated": calls sample_random_state of the underlying env object (errors if it has none)

    "conv_hull_img": samples an image based on the convex hull embeddings

    "None": no forgetting in case of SAC and for DQN this defaults to decaying 
            the values of actions that are not taken without state sampling

    "Object": If an actual world model is passed its "sample()" method is called


    In the paper we only use "generated" for ChooseBox and otherwise use "env".
    The other options are unused.

"""


class DQN_F(DQN):
    def __init__(
        self,
        eta_forgetting,
        baseline,
        forget_ratio=0.05,
        world_model=None,
        *args,
        **kwargs,
    ):

        if callable(baseline):
            self.baseline = baseline
        else:
            self.baseline = ConstantSchedule(baseline)

        if callable(forget_ratio):
            self.forget_ratio = forget_ratio
        else:
            self.forget_ratio = ConstantSchedule(forget_ratio)

        if callable(eta_forgetting):
            self.eta_forgetting = eta_forgetting
        else:
            self.eta_forgetting = ConstantSchedule(eta_forgetting)

        self.world_model = world_model
        self.custom_rng = np.random.default_rng()
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        # Baseline schedule
        baseline = self.baseline(self._current_progress_remaining)
        forget_ratio = self.forget_ratio(self._current_progress_remaining)
        eta_forgetting = self.eta_forgetting(self._current_progress_remaining)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)

                if self.world_model is None:
                    next_q_values_copy = copy.deepcopy(next_q_values)

                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values_action_taken = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values_action_taken, target_q_values)

            if (
                self.world_model == "conv_hull_img"
            ):  # convex hull sampling for image based observations e.g. Atari
                k_chull = 3  # Check
                num_forget = max([int(batch_size * forget_ratio), 1])

                # replay_data
                # Pick $k$ elements
                picked_imgs = replay_data.observations[
                    self.custom_rng.choice(self.batch_size, size=k_chull, replace=False)
                ]

                with th.no_grad():  # diff thru q net only
                    # encode imgs
                    encoded_imgs_hull = self.q_net.extract_features(
                        picked_imgs, self.q_net.features_extractor
                    )

                    # hull sampling
                    sampled_encodings = sample_hull(
                        encoded_imgs_hull, n=num_forget, rng=self.custom_rng
                    )

                forget_state_values = self.q_net.q_net(sampled_encodings)

                with th.no_grad():
                    forget_state_values_frozen = self.q_net.q_net(sampled_encodings)

                loss_forgetting = F.smooth_l1_loss(
                    forget_state_values,
                    (1 - eta_forgetting) * forget_state_values_frozen
                    + eta_forgetting
                    * baseline
                    * th.ones(num_forget, self.env.action_space.n, device=self.device),
                )

            else:
                if self.world_model is not None:
                    num_forget = max([int(batch_size * forget_ratio), 1])
                    if self.world_model == "env":
                        states = th.tensor(
                            [
                                self.env.observation_space.sample()
                                for _ in range(num_forget)
                            ],
                            device=self.device,
                        )
                    elif self.world_model == "generated":
                        states = th.tensor(
                            [
                                self.env.env_method("sample_random_state",rng=self.custom_rng)[0]
                                for _ in range(num_forget)
                            ],
                            device=self.device,
                        )
                    elif self.world_model == "stored_data":
                        bank = StateBank("reachability_results/" + self.env.spec.id   + "/bank")
                        obs, meta = bank.sample(256, by="room", rng=self.custom_rng) 

                        states = th.from_numpy(obs).to(self.device)
                        
                    else:
                        states = self.world_model.sample(num_forget)

                    forget_state_values = self.q_net(states)

                    with th.no_grad():
                        forget_state_values_frozen = self.q_net(states)

                    loss_forgetting = F.smooth_l1_loss(
                        forget_state_values,
                        (1 - eta_forgetting) * forget_state_values_frozen
                        + eta_forgetting
                        * baseline
                        * th.ones(
                            num_forget, self.env.action_space.n, device=self.device
                        ),
                    )

                else:
                    # Non-taken action value decay
                    possible_actions = np.broadcast_to(
                        list(range(self.env.action_space.n)),
                        (batch_size, self.env.action_space.n),
                    )

                    mask = ~(possible_actions == replay_data.actions.cpu())

                    q_values_not_taken_actions = current_q_values[mask].reshape(
                        batch_size, self.env.action_space.n - 1
                    )

                    target_q_values_not_taken_actions = next_q_values_copy[
                        mask
                    ].reshape(batch_size, self.env.action_space.n - 1)

                    loss_forgetting = F.smooth_l1_loss(
                        q_values_not_taken_actions,
                        (1 - self.eta_forgetting) * target_q_values_not_taken_actions
                        + self.eta_forgetting
                        * baseline
                        * th.ones(
                            batch_size, self.env.action_space.n - 1, device=self.device
                        ),
                    )

            loss = loss + loss_forgetting

            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

class SAC_F(SAC):
    def __init__(
        self,
        eta_forgetting,
        baseline,
        forget_ratio=0.05,
        world_model=None,
        *args,
        **kwargs,
    ):
        self.eta_forgetting = eta_forgetting
        if callable(baseline):
            self.baseline = baseline
        else:
            self.baseline = ConstantSchedule(baseline)
        self.world_model = world_model

        if callable(forget_ratio):
            self.forget_ratio = forget_ratio
        else:
            self.forget_ratio = ConstantSchedule(forget_ratio)

        if callable(eta_forgetting):
            self.eta_forgetting = eta_forgetting
        else:
            self.eta_forgetting = ConstantSchedule(eta_forgetting)

        self.custom_rng = np.random.default_rng()

        super().__init__(*args, **kwargs)

        if self.world_model == "stored_data":
            # For GoExplore-like generation of states (Ant!)
            self.bank = StateBank("reachability_results/" + self.env.envs[0].spec.id     + "/bank")



    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        # Update baseline acc. to schedule
        baseline = self.baseline(self._current_progress_remaining)
        forget_ratio = self.forget_ratio(self._current_progress_remaining)
        eta_forgetting = self.eta_forgetting(self._current_progress_remaining)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )

            # Compute critic loss
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)  # for type checker

            #### FORGETTING
            if self.world_model is not None:
                num_forget = max([int(batch_size * forget_ratio), 1])
                if self.world_model == "env":
                    states = th.tensor(
                        [
                            self.env.observation_space.sample()
                            for _ in range(num_forget)
                        ],
                        device=self.device,
                    )
                elif self.world_model == "stored_data":
                    obs, meta =  self.bank.sample(num_forget, by="room", rng=self.custom_rng) 

                    states = th.from_numpy(obs).to(self.device)
                else:
                    states = self.world_model.sample(num_forget)

                actions = th.tensor(
                    [self.action_space.sample() for _ in range(num_forget)],
                    device=self.device,
                )

                # states, actions

                forget_q_values = self.critic(states, actions)

                with th.no_grad():
                    # Compute the next Q-values using the target network
                    forget_q_values_frozen = self.critic_target(states, actions)

                critic_loss_forgetting = 0.5 * sum(
                    F.mse_loss(
                        forg_q_val,
                        (1 - eta_forgetting) * forg_q_val_frz
                        + eta_forgetting * baseline,
                    )
                    for forg_q_val, forg_q_val_frz in zip(
                        forget_q_values, forget_q_values_frozen
                    )
                )

            else:
                critic_loss_forgetting = th.zeros((), device=self.device)
            #####

            critic_losses.append(critic_loss.item() + critic_loss_forgetting.item())  # type: ignore[union-attr]

            critic_loss_total = critic_loss_forgetting + critic_loss

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss_total.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observations, state=None, episode_start=None, deterministic=True):
        # observations may come from a VecEnv: batch dimension = len(observations)
        batch = observations.shape[0] if hasattr(observations, "shape") else 1
        actions = [self.action_space.sample() for _ in range(batch)]
        return np.array(actions), state


####################################################################################################
# Limited Training Agents
class LimitedSAC(SAC):
    def __init__(
        self,
        max_training_steps=0,  # default: no training at all
        *args,
        **kwargs,
    ):
        self.max_training_steps = max_training_steps
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self.num_timesteps > self.max_training_steps:
            return
        else:
            super().train(gradient_steps=gradient_steps, batch_size=batch_size)


class LimitedDQN(DQN):
    def __init__(
        self,
        max_training_steps=0,  # default: no training at all
        *args,
        **kwargs,
    ):
        self.max_training_steps = max_training_steps
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self.num_timesteps > self.max_training_steps:
            return
        else:
            super().train(gradient_steps=gradient_steps, batch_size=batch_size)


class LimitedPPO(PPO):
    def __init__(
        self,
        max_training_steps=0,  # default: no training at all
        *args,
        **kwargs,
    ):
        self.max_training_steps = max_training_steps
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self.num_timesteps > self.max_training_steps:
            return
        else:
            super().train(gradient_steps=gradient_steps, batch_size=batch_size)
