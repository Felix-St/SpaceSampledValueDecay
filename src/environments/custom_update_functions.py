import ns_gym.base as base
from typing import Any


class BoundedDecrementUpdate(base.UpdateFn):
    r"""Decrement the probability of going in the intended direction by some k.

    Overview:

        .. math::
            Y_t = max(Y_{t-1} - k, C)

        where :math:`Y_t` is the parameter value at time step :math:`t` and :math:`k` is the amount to decrement the parameter by
        and :math:`C` is the lower bound of the parameter.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        k (float): The amount which the parameter is updated.
    """

    def __init__(self, scheduler, k, lo) -> None:
        super().__init__(scheduler)
        self.k = k
        self.lo = lo

    def _update(self, param, t) -> Any:
        if param >= self.lo:
            param -= self.k
        return param


class BoundedIncrementUpdate(base.UpdateFn):
    r"""Decrement the probability of going in the intended direction by some k.

    Overview:

        .. math::
            Y_t = max(Y_{t-1} - k, C)

        where :math:`Y_t` is the parameter value at time step :math:`t` and :math:`k` is the amount to decrement the parameter by
        and :math:`C` is the upper bound for the parameter.

    Args:
        scheduler (Type[base.Scheduler]): scheduler that determines when the update function fires.
        k (float): The amount which the parameter is updated.
    """

    def __init__(self, scheduler, k, hi) -> None:
        super().__init__(scheduler)
        self.k = k
        self.hi = hi

    def _update(self, param, t) -> Any:
        if param <= self.hi:
            param += self.k
        return param
