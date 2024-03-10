import functools as ft
import typing as tp
import warnings

import jax
import jax.experimental.ode as jode
import jax.numpy as jnp
import jax.tree_util as jtu

from .newton_raphson import NewtonRaphsonParams, newton_raphson


class ParametersIVP[T](tp.NamedTuple):
    """Parameters for the Initial Value Problem.

    Parameters:
        differential_operator: function to compute dynamics
        dt: step size for integration
        n_steps: total number of steps to integrate for
    """

    differential_operator: tp.Callable[[T], T]

    dt: float = 1e-3
    n_steps: int = 1000


type Integrator[T] = tp.Callable[[ParametersIVP[T], T], tuple[T, T]]


def _merge_states[T](preceding: T, final: T) -> T:
    return jtu.tree_map(
        lambda preceding, final: jnp.concatenate([preceding, jnp.expand_dims(final, 0)]), preceding, final
    )


def _adjoint_warning[T](fn: Integrator[T]) -> Integrator[T]:
    """Decorator to raise warning when adjoint-incompatible integrator used.

    Parameters:
        fn: integrator to be decorated

    Returns:
        decorated integrator -- raising warning when differentiated through
    """

    class AdjointWarning(UserWarning): ...

    @jtu.Partial(jax.custom_jvp, nondiff_argnums=(0,))
    def wrapped_fn(ivp_params: ParametersIVP[T], initial_state: T) -> tuple[T, T]:
        return fn(ivp_params, initial_state)

    @wrapped_fn.defjvp
    def wrapped_fn_jvp(ivp_params: ParametersIVP[T], primals: T, tangents: T) -> tuple[T, T]:
        warnings.warn(f'{fn.__name__} not compatible with adjoint', AdjointWarning, stacklevel=1)
        return tp.cast(tuple[T, T], jax.jvp(jtu.Partial(fn, ivp_params), primals, tangents))

    return ft.wraps(fn)(wrapped_fn)


# TODO >> @danielkelshaw
#         Should we work with flattened state here, simplifying to working with jnp arrays
#            This means we don't have to work with arbitrary PyTrees...


@_adjoint_warning
def euler_integrator[T](ivp_params: ParametersIVP[T], initial_state: T) -> tuple[T, T]:
    """Forward-Euler method for integration of the initial value problem.

    Parameters:
        ivp_params: parameters for the initial value problem
        initial_state: state at t=0

    Returns:
        final_state: final state of the initial value problem
        full_state: entire solution of the initial value problem
    """

    def _single_step(state: T, _: None) -> tuple[T, T]:
        update = ivp_params.differential_operator(state)
        next_state = jtu.tree_map(lambda x, dxdt: x + dxdt * ivp_params.dt, state, update)

        return next_state, state

    final_state, preceding_states = jax.lax.scan(_single_step, initial_state, None, int(ivp_params.n_steps))
    full_state = _merge_states(preceding=preceding_states, final=final_state)

    return final_state, full_state


@_adjoint_warning
def implicit_euler_integrator[T](ivp_params: ParametersIVP[T], initial_state: T) -> tuple[T, T]:
    """implicit-Euler method for integration of the initial value problem.

    Parameters:
        ivp_params: parameters for the initial value problem
        initial_state: state at t=0 to integrate from

    Returns:
        final_state: final state of the initial value problem
        full_state: entire solution of the initial value problem
    """

    # TODO >> @danielkelshaw
    #         Should we consider passing nr_params as an argument?
    nr_params = NewtonRaphsonParams(max_steps=1000, target_residual=1e-9)

    def _residual(curr_state: T, state: T) -> T:
        update = ivp_params.differential_operator(state)
        return jtu.tree_map(lambda s, cs, u: s - cs - u * ivp_params.dt, state, curr_state, update)

    def _single_step(state: T, _: None) -> tuple[T, T]:
        # initial guess for the newton-raphson is the forward-Euler method
        update = ivp_params.differential_operator(state)
        nr_initial_state = jtu.tree_map(lambda x, dxdt: x + dxdt * ivp_params.dt, state, update)

        # create partial residual for current time-step
        p_residual = jtu.Partial(_residual, state)

        # compute optimised state
        next_state, _ = newton_raphson(p_residual, nr_initial_state, nr_params)  # type: ignore

        return next_state, state

    final_state, preceding_states = jax.lax.scan(_single_step, initial_state, None, ivp_params.n_steps)
    full_state = _merge_states(preceding=preceding_states, final=final_state)

    return final_state, full_state


@_adjoint_warning
def rk4_integrator[T](ivp_params: ParametersIVP[T], initial_state: T) -> tuple[T, T]:
    """Runge-Kutta (4th order) method for integration of the initial value problem.

    Parameters:
        ivp_params: parameters for the initial value problem
        initial_state: state at t=0 to integrate from

    Returns:
        final_state: final state of the initial value problem
        full_state: entire solution of the initial value problem
    """

    def _single_step(state: T, _: None) -> tuple[T, T]:
        k1 = ivp_params.differential_operator(state)
        k2 = ivp_params.differential_operator(jtu.tree_map(lambda s, k: s + ivp_params.dt * k / 2.0, state, k1))
        k3 = ivp_params.differential_operator(jtu.tree_map(lambda s, k: s + ivp_params.dt * k / 2.0, state, k2))
        k4 = ivp_params.differential_operator(jtu.tree_map(lambda s, k: s + ivp_params.dt * k, state, k3))

        update = jtu.tree_map(lambda k1, k2, k3, k4: (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, k1, k2, k3, k4)

        next_state = jtu.tree_map(lambda x, dxdt: x + dxdt * ivp_params.dt, state, update)

        return next_state, state

    final_state, preceding_states = jax.lax.scan(_single_step, initial_state, None, ivp_params.n_steps)
    full_state = _merge_states(preceding=preceding_states, final=final_state)

    return final_state, full_state


def _timewrap[T](fn: tp.Callable[[T], T]) -> tp.Callable[[T, jax.Array], T]:
    """Injects time-dependence into time-independent function.

    Parameters:
        fn: time-independent function to wrap

    Returns:
        _fn: wrapped function, injecting time-dependence
    """

    @ft.wraps(fn)
    def _fn(x: T, t: jax.Array) -> T:
        """Over-ridden function with injected time-dependence.

        Parameters:
            x: Input state.
            t: Time variable.

        Returns:
            Original output of function, with no modification.
        """

        return fn(x)

    return _fn


def odeint[T](ivp_params: ParametersIVP[T], initial_state: T) -> tuple[T, T]:
    """DOPRI (4,5th order) method for integration of initial value problem -- adjoint compatible.

    Parameters:
        ivp_params: parameters for the initial value problem
        initial_state: state at t=0 to integrate from

    Returns:
        final_state: final state of the initial value problem
        full_state: entire solution of the initial value problem
    """

    differential_operator = _timewrap(ivp_params.differential_operator)
    t_record = jnp.linspace(0.0, ivp_params.dt * ivp_params.n_steps, ivp_params.n_steps + 1)

    full_state = jode.odeint(differential_operator, initial_state, t_record)
    final_state = jtu.tree_map(lambda x: x[-1], full_state)

    return final_state, full_state
