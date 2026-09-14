"""
Nikishin-style periodic resets for Stable-Baselines3 (DQN / SAC).

Reference: Nikishin et al., "The Primacy Bias in Deep Reinforcement Learning",
ICML 2022. Periodically re-initialize the agent's network(s) while PRESERVING
the replay buffer. For 3-layer MLPs ([256, 256] + head) the paper resets the
networks entirely, which is what this callback does.

The same file provides L2-to-init (regenerative regularization, Kumar et al.
2023) as `with_l2_init`, and the two compose:

    SAC_Reset  = with_resets(SAC)
    SAC_L2     = with_l2_init(SAC)
    SAC_Both   = with_resets(with_l2_init(SAC))
"""

from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN


class PeriodicResetCallback(BaseCallback):
    """Re-initialize the agent's networks every `reset_interval` env steps.

    Args:
        reset_interval: environment steps between resets. Aim for ~5-15 resets
            over a run (e.g. 1e5 for a 5e5-step run).
        rewarm_epsilon: DQN only. After a reset the network is random but the
            epsilon-greedy schedule sits at its floor, so exploration is
            effectively off. If True, epsilon is re-annealed from
            `exploration_initial_eps` to `exploration_final_eps` over
            `epsilon_warmup` steps after each reset. 
        epsilon_warmup: length of that re-anneal, in env steps.
        reset_at: optional explicit list of timesteps to reset at. Overrides
            `reset_interval` if given.
    """

    def __init__(
        self,
        reset_interval: int = 100_000,
        rewarm_epsilon: bool = False,
        epsilon_warmup: int = 10_000,
        reset_at: Optional[list] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.reset_interval = reset_interval
        self.rewarm_epsilon = rewarm_epsilon
        self.epsilon_warmup = epsilon_warmup
        self.reset_at = sorted(reset_at) if reset_at is not None else None
        self._last_reset = 0
        self._n_resets = 0

    def _on_training_start(self) -> None:
        self._install_epsilon_rewarm() # completely override stock schedule
        self._last_reset = self.model.num_timesteps

    def _due(self) -> bool:
        t = self.model.num_timesteps
        if self.reset_at is not None:
            return bool(self.reset_at) and t >= self.reset_at[0]
        return (t - self._last_reset) >= self.reset_interval

    def _on_step(self) -> bool:
        if self._due():
            if self.reset_at is not None:
                self.reset_at.pop(0)
            self._reset_agent()
            self._last_reset = self.model.num_timesteps
            self._n_resets += 1
        return True

    def _reset_agent(self) -> None:
        model = self.model

        # Stash the buffer: it must survive the reset. Shrinking buffer_size
        # first stops _setup_model from allocating a second full-size buffer
        buffer = model.replay_buffer
        buffer_size, model.buffer_size = model.buffer_size, 1

        # A fixed seed would make every re-initialization identical.
        seed, model.seed = model.seed, None

        model._setup_model()

        model.buffer_size = buffer_size
        model.seed = seed
        model.replay_buffer = buffer

        if self.rewarm_epsilon and hasattr(model, "exploration_schedule"):
            self._install_epsilon_rewarm()

        # Let mixins re-attach anything bound to the old policy/optimizers.
        if hasattr(model, "_after_reset"):
            model._after_reset()

        if self.verbose:
            print(f"[reset] step {model.num_timesteps} (reset #{self._n_resets + 1})")

    def _install_epsilon_rewarm(self) -> None:
        """Replace DQN's global epsilon schedule with one anchored at the reset."""
        if not isinstance(self.model, DQN): # not for SAC for example
            return 
        model = self.model
        start = model.exploration_initial_eps
        end = model.exploration_final_eps
        warmup = max(1, int(model.exploration_fraction * self.reset_interval))
        anchor = model.num_timesteps

        def schedule(_progress_remaining: float) -> float:
            frac = min(1.0, (model.num_timesteps - anchor) / warmup)
            return start + frac * (end - start)

        model.exploration_schedule = schedule



# ---------------------------------------------------------------------------
# Drop-in algorithm classes
# ---------------------------------------------------------------------------


class ResetMixin:
    """Mixin that makes an SB3 off-policy algorithm reset itself periodically.
 
        model = SAC_Reset("MlpPolicy", env, reset_interval=250_000)
        model.learn(total_timesteps=2_000_000)
 
    With `reset_interval=None` and no `reset_at`, no resets happen.
    """
 
    def __init__(
        self,
        *args,
        reset_interval: Optional[int] = None,
        rewarm_epsilon: bool = False,
        epsilon_warmup: int = 10_000,
        reset_at: Optional[list] = None,
        **kwargs,
    ):
        self._reset_cfg = dict(
            reset_interval=reset_interval,
            rewarm_epsilon=rewarm_epsilon,
            epsilon_warmup=epsilon_warmup,
            reset_at=reset_at,
        )
        super().__init__(*args, **kwargs)
 
    def learn(self, total_timesteps, callback=None, **kwargs):
        cfg = dict(self._reset_cfg)
        if cfg["reset_interval"] is not None or cfg["reset_at"] is not None:
            if cfg["reset_interval"] is None:
                cfg["reset_interval"] = 10**18
            reset_cb = PeriodicResetCallback(
                verbose=getattr(self, "verbose", 0), **cfg
            )
            if callback is None:
                callback = reset_cb
            elif isinstance(callback, list):
                callback = callback + [reset_cb]
            else:
                callback = [callback, reset_cb]
        return super().learn(total_timesteps, callback=callback, **kwargs)



def with_resets(base_cls, name: Optional[str] = None):
    """Build a reset-enabled version of any SB3 off-policy algorithm class."""
    return type(name or f"{base_cls.__name__}_Reset", (ResetMixin, base_cls), {})



# ---------------------------------------------------------------------------
# L2-to-init (regenerative regularization, Kumar et al. 2023)
# ---------------------------------------------------------------------------

def _policy_optimizers(model):
    """Optimizers that update the online networks (not the entropy coef)."""
    policy = model.policy
    opts = []
    for holder in (policy, getattr(policy, "actor", None), getattr(policy, "critic", None)):
        opt = getattr(holder, "optimizer", None) if holder is not None else None
        if opt is not None and not any(opt is o for o in opts):
            opts.append(opt)
    return opts


class L2InitMixin:
    """Pull weights toward their initialization with an L2 penalty.

    Adds lambda * (theta - theta_0) to the gradient before every optimizer
    step, which is exactly the gradient of a lambda/2 * ||theta - theta_0||^2
    term added to the loss (the factor of 2 is absorbed into lambda, so
    compare magnitudes with that in mind).

    Implemented by wrapping the optimizers' `step()` rather than rewriting
    `train()`, so it applies to whatever loss the base algorithm computes --
    including your SsVD variants -- and stays version-agnostic.

    Args:
        l2_lambda: regularization strength. Sweep on a log grid, e.g.
            {1e-5, 1e-4, 1e-3, 1e-2}; the useful range is problem-dependent.
        l2_anchor: "init" for L2-to-init (regenerative regularization), or
            "zero" for plain L2 / weight decay. The "zero" setting gives you
            the weight-decay control at no extra implementation cost, and is
            worth one config: it separates "pulling toward a fixed reference"
            from "pulling toward the initialization specifically".

    Target networks are excluded automatically -- they are not in any
    optimizer's parameter groups.
    """

    def __init__(self, *args, l2_lambda: float = 1e-4, l2_anchor: str = "init", **kwargs):
        self.l2_lambda = l2_lambda
        self.l2_anchor = l2_anchor
        super().__init__(*args, **kwargs)
        self._install_l2_init()

    def _install_l2_init(self) -> None:
        if self.l2_lambda == 0.0:
            return
        lam = self.l2_lambda
        to_zero = self.l2_anchor == "zero"

        for opt in _policy_optimizers(self):
            anchors = {}
            for group in opt.param_groups:
                for p in group["params"]:
                    anchors[p] = None if to_zero else p.detach().clone()

            def make_step(opt=opt, anchors=anchors):
                orig_step = opt.step

                def step(*args, **kwargs):
                    for group in opt.param_groups:
                        for p in group["params"]:
                            if p.grad is None or p not in anchors:
                                continue
                            anchor = anchors[p]
                            delta = p.detach() if anchor is None else p.detach() - anchor
                            p.grad.add_(delta, alpha=lam)
                    return orig_step(*args, **kwargs)

                return step

            opt.step = make_step()

    def _after_reset(self) -> None:
        """Re-anchor and re-wrap after a Nikishin-style reset.

        The optimizers are new objects and the network has a new
        initialization, so the anchor is re-snapshotted -- which is the
        intended semantics: L2-to-init means the *current* init.
        """
        self._install_l2_init()
        parent = getattr(super(), "_after_reset", None)
        if parent is not None:
            parent()


def with_l2_init(base_cls, name=None):
    """Build an L2-to-init version of any SB3 off-policy algorithm class."""
    return type(name or f"{base_cls.__name__}_L2Init", (L2InitMixin, base_cls), {})





if __name__ == "__main__":
    import gymnasium as gym
    from stable_baselines3 import DQN

    env = gym.make("CartPole-v1")
    DQN_Reset = with_resets(DQN)

    env = gym.make("CartPole-v1")
    model = DQN_Reset(
        "MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256]), verbose=1
    )
    model.learn(total_timesteps=60_000, reset_interval=20_000)