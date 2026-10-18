"""SIGReg: sliced Epps–Pulley test of embeddings against N(0, I).

Re-implemented from the paper's description (not from the official code):
Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning
Without the Heuristics", arXiv:2511.08544.

For samples `x_1..x_N` of a 1D variable, the Epps–Pulley statistic compares the
empirical characteristic function `phi_N(t) = mean_n exp(i t x_n)` with the target
`phi(t) = exp(-t^2 / 2)` of N(0, 1):

    EP = N * integral |phi_N(t) - phi(t)|^2 w(t) dt,   w(t) = exp(-t^2 / 2).

SIGReg projects the d-dimensional embeddings onto random unit directions `a`
(resampled every call), computes EP on each projection `a^T z` and averages over
directions. The integrand is even in `t`, so the integral over `[-t_max, t_max]` is
twice a trapezoidal rule on `num_points` equally spaced nodes in `[0, t_max]`.

Under the null (i.i.d. N(0, I) samples) `E[N |phi_N(t) - phi(t)|^2] = 1 - exp(-t^2)`
exactly, so the expected statistic (its "noise floor") does not depend on `N` or
`d`; see `SIGReg.noise_floor`. Over the whole real line it equals
`sqrt(2 pi) - sqrt(2 pi / 3) ~= 1.059`.
"""

import torch


class SIGReg:
    """Sliced Epps–Pulley statistic against N(0, I).

    Attributes:
        num_slices: Number of random unit directions per call.
        num_points: Number of quadrature nodes on `[0, t_max]`.
        t_max: Upper limit of the (symmetric) integration range.
    """

    def __init__(
        self,
        num_slices: int = 1024,
        num_points: int = 17,
        t_max: float = 3.0,
        generator: torch.Generator | None = None,
    ) -> None:
        """Initializes the statistic.

        Args:
            num_slices: Number of random unit directions, resampled every call.
            num_points: Number of quadrature nodes (at least 2).
            t_max: Integration range is `[-t_max, t_max]`; the weight
                `exp(-t^2 / 2)` makes the tail beyond 3 negligible.
            generator: Optional RNG for the directions (must live on the same
                device as the embeddings); defaults to the global RNG.

        Raises:
            ValueError: An argument is out of range.
        """
        if num_slices < 1 or num_points < 2 or t_max <= 0:
            raise ValueError("need num_slices >= 1, num_points >= 2 and t_max > 0")
        self.num_slices = num_slices
        self.num_points = num_points
        self.t_max = t_max
        self._generator = generator

    def _grid(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the nodes `t` and the combined weights `2 * trapezoid * w(t)`."""
        t = torch.linspace(0.0, self.t_max, self.num_points, device=device)
        step = self.t_max / (self.num_points - 1)
        trapezoid = torch.full_like(t, step)
        trapezoid[0] = trapezoid[-1] = step / 2
        return t, 2.0 * trapezoid * torch.exp(-0.5 * t.square())

    @property
    def noise_floor(self) -> float:
        """Expected value of the statistic on i.i.d. N(0, I) samples."""
        t, weights = self._grid(torch.device("cpu"))
        t64, weights64 = t.double(), weights.double()
        return float((weights64 * (1.0 - torch.exp(-t64.square()))).sum())

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """Computes the statistic, averaged over random directions.

        Args:
            z: Samples, shape `(N, d)`. Computation is in fp32 regardless of the
                input dtype or an enclosing autocast region.

        Returns:
            Scalar tensor (differentiable with respect to `z`).
        """
        if z.ndim != 2:
            raise ValueError(f"z must be (N, d), got {tuple(z.shape)}")
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            n, d = z.shape
            directions = torch.randn(
                d, self.num_slices, device=z.device, generator=self._generator
            )
            directions = directions * directions.square().sum(0, keepdim=True).rsqrt()
            t, weights = self._grid(z.device)
            # (N, slices, points): phase t * a^T z_n for every sample/slice/node.
            phase = torch.matmul(z, directions)[:, :, None] * t
            ecf_real = torch.cos(phase).mean(dim=0)
            ecf_imag = torch.sin(phase).mean(dim=0)
            target = torch.exp(-0.5 * t.square())
            sq_err = (ecf_real - target).square() + ecf_imag.square()
            per_slice = n * (sq_err * weights).sum(dim=-1)
            return per_slice.mean()
