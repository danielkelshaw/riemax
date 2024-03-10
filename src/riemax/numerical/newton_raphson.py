from __future__ import annotations

import typing as tp

import jax
import jax._src.flatten_util as jfu
import jax.numpy as jnp
import jax.scipy as jsp
import opt_einsum as oe


class NewtonRaphsonParams(tp.NamedTuple):
    max_steps: int = 100
    min_steps: int = 2

    damping_factor: float = 1.0
    target_residual: float = 1e-7


class _NewtonRaphsonState[T](tp.NamedTuple):
    flat: jax.Array
    step: int
    residual: float


class NewtonConvergenceState(tp.NamedTuple):
    """Store for information about convergence of Newton method.

    Parameters:
        step: step on which the newton method stopped
        max_steps: maximum number of steps the newton method was allowed to take
        residual: residual of the convergence
        target_residual: user-specified tolerance for conversion
    """

    step: int
    max_steps: int

    residual: float
    target_residual: float

    @classmethod
    def _from_params_state(
        cls, nr_params: NewtonRaphsonParams, nr_state: _NewtonRaphsonState
    ) -> NewtonConvergenceState:
        return cls(
            step=nr_state.step,
            max_steps=nr_params.max_steps,
            residual=nr_state.residual,
            target_residual=nr_params.target_residual,
        )

    @property
    def converged(self) -> bool:
        """Determines whether Newton method converged for given budget.

        Returns:
            whether the newton method converged
        """

        step_limit_exceeded = self.step > self.max_steps
        residual_converged = self.residual < self.target_residual

        return residual_converged | step_limit_exceeded


def newton_raphson[T](
    fn_residual: tp.Callable[[T], T], initial_guess: T, nr_parameters: NewtonRaphsonParams | None = None
) -> tuple[T, NewtonConvergenceState]:
    """Newton-Raphson root finding for arbitrary PyTrees.

    Parameters:
        fn_residual: function to compute the residual you are trying to minimise
        initial_guess: starting point for the optimisation procedure
        nr_parameters: parameters for use in the optimisation process

    Returns:
        optimised_state: optimised state which minimises the given residual function
        nr_convergence: auxiliary information about state of the optimisation process
    """

    if not nr_parameters:
        nr_parameters = NewtonRaphsonParams()

    flat, unflatten = jfu.ravel_pytree(initial_guess)

    def curried_fn_residual(z: jax.Array) -> jax.Array:
        return jfu.ravel_pytree(fn_residual(unflatten(z)))[0]

    initial_nr_state = _NewtonRaphsonState(flat=flat, step=0, residual=float(jnp.inf))

    def _condition(nr_state: _NewtonRaphsonState) -> bool:
        # step size tests
        at_least_min_steps = nr_state.step < nr_parameters.min_steps
        step_okay = nr_state.step < nr_parameters.max_steps

        # precision test
        not_converged = nr_state.residual > nr_parameters.target_residual

        return at_least_min_steps | (step_okay & not_converged)

    def _body_fn(nr_state: _NewtonRaphsonState) -> _NewtonRaphsonState:
        # compute the residual and jacobian
        rx = curried_fn_residual(nr_state.flat)
        rx_jacobian = jax.jacobian(curried_fn_residual)(nr_state.flat)

        # solve the system of equations and update
        diff = jsp.linalg.solve(rx_jacobian, rx)
        flat = nr_state.flat - nr_parameters.damping_factor * diff

        # update auxiliary variables
        step = nr_state.step + 1

        # compute rms_norm of residual
        f_rx: float = oe.contract('... -> ', rx**2) / rx.shape[0]

        return _NewtonRaphsonState(flat=flat, step=step, residual=f_rx)

    optimised_nr_state = jax.lax.while_loop(_condition, _body_fn, initial_nr_state)
    optimised_state = unflatten(optimised_nr_state.flat)

    nr_convergence = NewtonConvergenceState._from_params_state(nr_params=nr_parameters, nr_state=optimised_nr_state)

    return optimised_state, nr_convergence
