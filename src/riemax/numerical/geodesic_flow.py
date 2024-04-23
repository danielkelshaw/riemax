import typing as tp

import jax
import jax.tree_util as jtu
from riemax.numerical.integrators import ParametersIVP, euler_integrator

from ..manifold.operators import grad
from ..manifold.types import M, MetricFn, TpM

type DistanceFn = tp.Callable[[jax.Array, jax.Array], jax.Array]


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
