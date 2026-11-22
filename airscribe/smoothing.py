"""One Euro filter for the fingertip position.

A low-pass filter whose cutoff rises with speed: when the finger is still or
slow it smooths heavily (kills jitter); when it moves fast the cutoff opens up
so corners and quick strokes are not rounded off or lagged.
"""
import math
from typing import Optional, Tuple


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter1D:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x_prev: Optional[float] = None
        self.dx_prev = 0.0
        self.t_prev: Optional[float] = None

    def __call__(self, x: float, t: float) -> float:
        if self.x_prev is None or self.t_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class PointSmoother:
    """Applies a One Euro filter independently to x and y."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.fx = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter1D(min_cutoff, beta, d_cutoff)

    def __call__(self, p: Tuple[float, float], t: float) -> Tuple[float, float]:
        return self.fx(p[0], t), self.fy(p[1], t)

    def reset(self) -> None:
        """Call when tracking is lost so the next position starts fresh
        instead of gliding in from where the hand used to be."""
        self.fx.reset()
        self.fy.reset()
