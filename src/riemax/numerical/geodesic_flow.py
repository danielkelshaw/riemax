import typing as tp

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ..manifold.operators import grad
from ..manifold.types import M, MetricFn, TangentSpace, TpM
from ..numerical.integrators import ParametersIVP, _merge_states, euler_integrator

type DistanceFn = tp.Callable[[jax.Array, jax.Array], jax.Array]
type GeodesicFlow = tp.Callable[[jax.Array, jax.Array], jax.Array]
type GeodesicInterpolator = tp.Callable[[jax.Array, jax.Array], TangentSpace[jax.Array]]


def _euler_interpolator_integrator[T](ivp_params: ParametersIVP[T], initial_state: T) -> TangentSpace:
    def _single_step(state: T, _: None) -> tuple[T, tuple[T, T]]:
        update = ivp_params.differential_operator(state)
        next_state = jax.tree.map(lambda x, dxdt: x + dxdt * ivp_params.dt, state, update)

        return next_state, (state, update)

    final_state, preceding_states = jax.lax.scan(_single_step, initial_state, None, int(ivp_params.n_steps))
    final_update = ivp_params.differential_operator(final_state)

    full_state = _merge_states(preceding=preceding_states, final=(final_state, final_update))

    return TangentSpace(*full_state)


def construct_geodesic_flow(fn_distance: DistanceFn, fn_metric: MetricFn) -> GeodesicFlow:
    """Produce a function capable of computing the geodesic flow

    Parameters:
        fn_distance: distance function defined on the manifold.
        fn_metric: metric function used to define the manifold

    Returns:
        fn_geodesic_flow: function to compute value of geodesic flow.
    """

    def fn_geodesic_flow(p: jax.Array, q: jax.Array) -> jax.Array:
        """Compute geodesic flow

        Parameters:
            p: point on the manifold at which to source the vector field
            q: point on the manifold to evaluate the resulting vector field

        Returns:
            evaluation of the geodesic flow
        """

        def distance_from_p(x: jax.Array) -> jax.Array:
            return fn_distance(p, x)

        return grad(distance_from_p, metric=fn_metric)(q)

    return fn_geodesic_flow


def construct_geodesic_interpolator(fn_distance: DistanceFn, fn_geodesic_flow: GeodesicFlow) -> GeodesicInterpolator:
    ts_eval = jnp.expand_dims(jnp.linspace(0.0, 1.0, 1001, endpoint=True), 1)

    def _interpolate(p: jax.Array, q: jax.Array) -> TangentSpace[jax.Array]:
        d_pq = fn_distance(p, q)

        def _dynamics(x: jax.Array):
            return -fn_geodesic_flow(p, x) * d_pq

        ivp_params = ParametersIVP(_dynamics)
        interpolant = _euler_interpolator_integrator(ivp_params, q)

        return interpolant

    def merge_interpolants(gamma_one, gamma_two):
        return ts_eval * gamma_one + (1.0 - ts_eval) * gamma_two

    def fn_interpolate(p: jax.Array, q: jax.Array) -> TangentSpace[jax.Array]:
        gamma_qp = jax.tree.map(lambda x: jnp.flip(x, 0), _interpolate(p, q))
        gamma_qp = TangentSpace(point=gamma_qp.point, vector=-gamma_qp.vector)

        gamma_pq = _interpolate(q, p)

        return jax.tree.map(merge_interpolants, gamma_qp, gamma_pq)

    return fn_interpolate


def construct_geodesic_field(
    q: M[jax.Array], fn_distance: DistanceFn, fn_metric: MetricFn
) -> tp.Callable[[M[jax.Array]], TpM[jax.Array]]:
    fn_dist = jtu.Partial(fn_distance, q)

    def geodesic_field(p: M[jax.Array]) -> TpM[jax.Array]:
        return grad(fn_dist, fn_metric)(p)

    return geodesic_field


def connecting_geodesic(p: M[jax.Array], q: M[jax.Array], dt: float, fn_distance: DistanceFn, fn_metric: MetricFn):
    fn_geodesic_field = construct_geodesic_field(p, fn_distance, fn_metric)

    def dynamics(r: M[jax.Array]) -> TpM[jax.Array]:
        return -fn_geodesic_field(r)

    geodesic_length = fn_distance(p, q)
    n_steps = int(geodesic_length // dt) + 1

    ivp_params = ParametersIVP(dynamics, dt, n_steps)
    return euler_integrator(ivp_params, q)
