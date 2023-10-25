import functools as ft
import typing as tp

import jax
import jax.tree_util as jtu

from ..numerical.integrators import Integrator, ParametersIVP
from ..numerical.newton_raphson import NewtonRaphsonParams, newton_raphson
from .geodesic import geodesic_dynamics
from .types import M, MetricFn, TangentSpace, TpM

type ExponentialMap = tp.Callable[[TangentSpace[jax.Array]], tuple[M[jax.Array], TangentSpace[jax.Array]]]
type LogMap[*Ts] = tp.Callable[
    [TangentSpace[jax.Array] | M[jax.Array], M[jax.Array], *Ts], tuple[TangentSpace[jax.Array], bool]
]


def _transform_integrator_to_exp[T](integrator_output: tuple[TangentSpace[T], TangentSpace[T]]):
    final_state, full_state = integrator_output
    return final_state.point, full_state


def _integrator_to_exp[T](
    fn: tp.Callable[[TangentSpace[T]], tuple[TangentSpace[T], TangentSpace[T]]]
) -> tp.Callable[[TangentSpace[T]], tuple[M[T], TangentSpace[T]]]:
    @ft.wraps(fn)
    def _fn(state: TangentSpace[T]) -> tuple[M[T], TangentSpace[T]]:
        return _transform_integrator_to_exp(fn(state))

    return _fn


def exponential_map_factory(
    integrator: Integrator[TangentSpace[jax.Array]], dt: float, metric: MetricFn, n_steps: int | None = None
) -> ExponentialMap:
    r"""Produce an exponential map, $\exp: TM \rightarrow M$.

    !!! note "Example"

        ```python
        # ...

        exp_map = exponential_map_factory(riemax.numerical.integrators.odeint, dt=1e-3, metric=fn_metric)
        ```

    Parameters:
        integrator: choice of integrator used to propgate dynamics
        dt: time-step for the integration
        metric: function to compute metric tensor
        n_steps: number of steps to integrate for

    Returns:
        exp_map: function for computing exponential map
    """

    if not n_steps:
        n_steps = int(1.0 // dt)

    dynamics = jtu.Partial(geodesic_dynamics, metric=metric)
    ivp_params = ParametersIVP(differential_operator=dynamics, dt=dt, n_steps=n_steps)

    @_integrator_to_exp
    def exp_map(state: TangentSpace[jax.Array]) -> tuple[TangentSpace[jax.Array], TangentSpace[jax.Array]]:
        return integrator(ivp_params, state)

    return exp_map


def shooting_log_map_factory(exp_map: ExponentialMap, nr_parameters: NewtonRaphsonParams | None = None) -> LogMap:
    r"""Produce log map, computed using a shooting method.

    !!! note "Efficacy of Shooting Solvers:"

        Shooting solvers typically require a good initial guess. If the initial guess for the velocity vector is too far
        from a true solution, this can tend to fail. We also note that, this does not guarantee obtaining the velocity
        vector of the globally length-minimising geodesic -- only of a valid geodesic which connects the two points.

    Parameters:
        exp_map: function used to compute the exponential map
        nr_parameters: parameters used in the Newton-Raphson optimisation

    Returns:
        log_map: function to compute the log map between $p, q \in M$
    """

    def log_map(
        p: TangentSpace[jax.Array] | M[jax.Array],
        q: M[jax.Array],
    ) -> tuple[TangentSpace[jax.Array], bool]:
        """Compute the log map between points p and q.

        Parameters:
            p: origin point on the manifold
            q: destination point on the manifold
            initial_p0_dot: initial guess for the tangent vector

        Returns:
            state which, when the exponential map is taken at p, yields q
        """

        if not isinstance(p, TangentSpace):
            p = TangentSpace(point=p, vector=(q - p))

        def shooting_residual(p_dot: TpM[jax.Array]) -> TpM[jax.Array]:
            initial_state = TangentSpace[jax.Array](point=p.point, vector=p_dot)
            point, _ = exp_map(initial_state)

            return point - q

        # root-finding for shooting residual
        p_dot, newton_convergence_state = newton_raphson(
            shooting_residual, initial_guess=p.vector, nr_parameters=nr_parameters
        )

        initial_condition = TangentSpace[jax.Array](point=p.point, vector=p_dot)

        return initial_condition, newton_convergence_state.converged

    return log_map
