from __future__ import annotations

import typing as tp

import einops
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ..numerical.integral_approximators import IntegralApproximationFn, mean_integration
from ._marked import manifold_marker
from .geometry import contravariant_metric_tensor, inner_product, sk_christoffel
from .types import MetricFn, TangentSpace


@manifold_marker.mark(jittable=True)
def geodesic_dynamics(state: TangentSpace[jax.Array], metric: MetricFn) -> TangentSpace[jax.Array]:
    r"""Compute update step for the geodesic dynamics.

    The geodesic equation

    $$
    \ddot{\gamma}^k + \Gamma^k_{\phantom{k}ij} \dot{\gamma}^i \dot{\gamma}^j  = 0
    $$

    is a second order ordinary differential equation. We take the conventional approach of splitting
    this into two first-order ordinary differential equations.

    Parameters:
        state: current state of the geodesic integration
        metric: covariant metric tensor used

    Returns:
        derivatives used to compute update for the state
    """

    sk_christ = sk_christoffel(state.point, metric)
    vector_dot = -jnp.einsum('kij, i, j -> k', sk_christ, state.vector, state.vector)

    return TangentSpace[jax.Array](state.vector, vector_dot)


def alternative_geodesic_dynamics(state: TangentSpace[jax.Array], metric: MetricFn) -> TangentSpace[jax.Array]:
    r"""Compute geodesic dynamics, as per (Arvanitidis, G., Hansen, LK., Hauberg, S., 2018).[^1]

    !!! note "Latent Space Oddity Approach"
        The paper 'Latent Space Oddity' provides a different formulation of the geodesic equation. It is not clear why
        this is useful or necessary, and obscures computation of the Christoffel symbols; nevertheless, an
        implementation is provided below.

        Interestingly, seminal papers: 'A Geometric take on Metric Learning', 'Metrics for Probabilistic Models' use a
        similar approach but are missing a term. It appears that these are incorrect and should likely be revised to
        reflect their errors.

        While the mathematical specification for the alternative dynamics makes use of `vec`, I avoid this to ensure we
        only have to compute the Jacobian of the metric tensor a single time.

    [^1]:
        Arvanitidis, Georgios, Lars Kai Hansen, and Søren Hauberg. ‘Latent Space Oddity: On the Curvature of Deep Generative Models’. arXiv, 2021. <a href="http://arxiv.org/abs/1710.11379">http://arxiv.org/abs/1710.11379</a>

    Parameters:
        state: current state of the geodesic integration
        metric: covariant metric tensor used

    Returns:
        derivatives used to compute update for the state
    """

    contra_g_ij = contravariant_metric_tensor(state.point, metric)

    dgdx = jax.jacobian(metric)(state.point)
    central_term = 2.0 * einops.rearrange(dgdx, 'i j k -> i (j k)') - einops.rearrange(dgdx, 'i j k -> k (i j)')

    vector_dot = -0.5 * contra_g_ij @ central_term @ jnp.kron(state.vector, state.vector)

    return TangentSpace[jax.Array](state.vector, vector_dot)


def _compute_discretised_energy(geodesic: TangentSpace[jax.Array], metric: MetricFn) -> jax.Array:
    return 0.5 * inner_product(geodesic.point, geodesic.vector, geodesic.vector, metric)


def _compute_discretised_length(geodesic: TangentSpace[jax.Array], metric: MetricFn) -> jax.Array:
    return jnp.sqrt(inner_product(geodesic.point, geodesic.vector, geodesic.vector, metric))


def _integrate_curve_quantity(
    fn_quantity: tp.Callable[[TangentSpace[jax.Array]], jax.Array],
    geodesic: TangentSpace[jax.Array],
    dt: float,
    integral_approximator: IntegralApproximationFn,
) -> jax.Array:
    return integral_approximator(jax.vmap(fn_quantity)(geodesic), dt)


def compute_geodesic_length(
    geodesic: TangentSpace[jax.Array],
    dt: float,
    metric: MetricFn,
    integral_approximator: IntegralApproximationFn = mean_integration,
) -> jax.Array:
    r"""Compute length of the geodesic.

    The length of a geodesic is defined as

    $$
    L(\gamma, \dot{\gamma}) = \int_0^1 \sqrt{ g_{\gamma} (\dot{\gamma}, \dot{\gamma}) } dt.
    $$

    We note that this is not necessarily equivalent to the geodesic distance between two points.

    Parameters:
        geodesic: point on the geodesic
        metric: function to compute the metric tensor

    Returns:
        length of the geodesic
    """

    quantity_fn = jtu.Partial(_compute_discretised_length, metric=metric)
    return _integrate_curve_quantity(quantity_fn, geodesic, dt, integral_approximator)


def compute_geodesic_energy(
    geodesic: TangentSpace[jax.Array],
    dt: float,
    metric: MetricFn,
    integral_approximator: IntegralApproximationFn = mean_integration,
) -> jax.Array:
    r"""Compute energy of the geodesic.

    The energy of a geodesic is defined as

    $$
    E(\gamma, \dot{\gamma}) = \int_0^1 g_{\gamma} (\dot{\gamma}, \dot{\gamma}) dt.
    $$

    Parameters:
        geodesic: point on the geodesic
        metric: function to compute the metric tensor

    Returns:
        energy of the geodesic
    """

    quantity_fn = jtu.Partial(_compute_discretised_energy, metric=metric)
    return _integrate_curve_quantity(quantity_fn, geodesic, dt, integral_approximator)
