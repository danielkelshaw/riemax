from __future__ import annotations

import functools as ft
import typing as tp

import einops
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from scipy.integrate import solve_bvp

from ..numerical.curves import CubicSpline
from ..numerical.integral_approximators import IntegralApproximationFn, mean_integration
from ._marked import manifold_marker
from .geometry import contravariant_metric_tensor, inner_product, sk_christoffel
from .types import M, MetricFn, TangentSpace


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
        metric: function defining the metric tensor on the manifold

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
        metric: function defining the metric tensor on the manifold

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
        metric: function defining the metric tensor on the manifold

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
        metric: function defining the metric tensor on the manifold

    Returns:
        energy of the geodesic
    """

    quantity_fn = jtu.Partial(_compute_discretised_energy, metric=metric)
    return _integrate_curve_quantity(quantity_fn, geodesic, dt, integral_approximator)


def minimising_geodesic(
    p: M[jax.Array],
    q: M[jax.Array],
    metric: MetricFn,
    optimiser: optax.GradientTransformation,
    num_nodes: int = 20,
    n_collocation: int = 100,
    iterations: int = 100,
    tol: float = 1e-4,
) -> tuple[TangentSpace[jax.Array], bool]:
    """Obtain energy-minimising geodesics between two points.

    This implementation models the geodesic as a cubic spline, constrained at the two end-points. An optimisation
    problem is solved, obtaining parameters of the cubic spline which minimise the energy of the resulting geodesic;
    ideally, obtaining the length-minimising geodesic between the two points.

    Parameters:
        p: first end-point of the geodesic
        q: second end-point of the geodesic
        metric: function defining the metric tensor on the manifold
        optimiser: optimiser to use for the optimisation procedure
        num_nodes: number of nodes to parameterise the cubic spline by
        n_collocation: number of points to evaluate energy at
        iterations: number of iterations to optimise for
        tol: tolerance for gradients of updates

    Returns:
        optimised geodesic, connecting the two points
        whether the optimisation procedure converged
    """

    curve = CubicSpline.from_nodes(p, q, num_nodes)
    tt = jnp.linspace(0, 1, n_collocation, endpoint=True)

    @jax.jit
    def loss_fn(params: jax.Array) -> jax.Array:
        p_fn = jtu.Partial(_compute_discretised_energy, metric=metric)
        discrete_fn_eval = jax.vmap(p_fn)(curve.evaluate(tt, params))

        return jnp.mean(discrete_fn_eval)

    @jax.jit
    def update(
        params: jax.Array, opt_state: optax.OptState
    ) -> tuple[jax.Array, tuple[jax.Array, optax.OptState, jax.Array]]:
        loss, grads = jax.value_and_grad(loss_fn)(params)
        max_grad = jnp.max(grads)

        updates, opt_state = optimiser.update(grads, opt_state, params=params)
        params = tp.cast(jax.Array, optax.apply_updates(params, updates))

        return loss, (params, opt_state, max_grad)

    params = curve.init_params()
    opt_state = optimiser.init(params)

    max_grad = jnp.inf
    for _ in range(iterations):
        loss, (params, opt_state, max_grad) = update(params, opt_state)

        if max_grad < tol:
            break

    geodesic = curve.evaluate(tt, params)

    return geodesic, max_grad < tol


def scipy_bvp_geodesic(
    p: jax.Array,
    q: jax.Array,
    metric: MetricFn,
    n_collocation: int = 100,
    explicit_jacobian: bool = False,
    tol: float = 1e-4,
) -> tuple[TangentSpace[jax.Array], bool]:
    """Obtain geodesic connecting two points using scipy.integrate.solve_bvp

    This method mirrors `minimising_geodesic` as scipy uses a similar scheme to solve boundary value problems. The scipy
    implementation does not consider fixed end-points though, and a separate set of boundary conditions must be
    optimised for. While the scipy implementation is more complete in terms of implementation, external calls are slower
    and cannot be jitted. The necessity for minimising a boundary condition residual is also a consideration.

    Parameters:
        p: first end-point of the geodesic
        q: second end-point of the geodesic
        metric: function defining the metric tensor on the manifold
        n_collocation: number of points to evaluate energy at
        explicit_jacobian: whether to use jacobian computed by jax
        tol: tolerance for gradients of updates

    Returns:
        optimised geodesic, connecting the two points
        whether the optimisation procedure converged
    """

    ndim = p.size
    dynamics = jtu.Partial(geodesic_dynamics, metric=metric)

    def numpy_wrapped(fn):
        @ft.wraps(fn)
        def inner(*args: np.ndarray):
            return tuple(map(np.asarray, fn(*map(jnp.asarray, args))))

        return inner

    def t_last(fn):
        @ft.wraps(fn)
        def inner(*args):
            return einops.rearrange(fn(*args), 't ... -> ... t')

        return inner

    def _fn_ode(_: jax.Array, x: jax.Array):
        geodesic_state = TangentSpace(point=x[:ndim, :].T, vector=x[ndim:, :].T)
        geodesic_update = jax.vmap(dynamics)(geodesic_state)

        geodesic_update = jnp.concatenate(jtu.tree_leaves(geodesic_update))

        return geodesic_update

    fn_ode = numpy_wrapped(t_last(jax.vmap(_fn_ode, in_axes=(None, 1))))

    fn_jacobian = None
    if explicit_jacobian:
        fn_jacobian = numpy_wrapped(t_last(jax.vmap(jax.jacobian(_fn_ode, argnums=1), in_axes=(None, 1))))

    @numpy_wrapped
    def fn_bc(ya, yb):
        ra = ya[:ndim] - p
        rb = yb[:ndim] - q

        return jnp.concatenate((ra, rb))

    x_init = np.linspace(0.0, 1.0, n_collocation, endpoint=True)

    gamma_init = np.einsum('i, j -> ij', p, (1.0 - x_init)) + np.einsum('i, j -> ij', q, x_init)
    gamma_dot_init = einops.repeat(q - p, 'i -> i t', t=n_collocation)

    y_init = np.concatenate((gamma_init, gamma_dot_init), axis=0)

    bvp_result = solve_bvp(fn_ode, fn_bc, x_init, y_init, fun_jac=fn_jacobian, tol=tol)
    geodesic = TangentSpace(point=bvp_result[:ndim, :].T, vector=bvp_result[ndim:, :].T)

    return geodesic, bvp_result.success
