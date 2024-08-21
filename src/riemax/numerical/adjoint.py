import functools as ft

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.extend import linear_util as lu
from jax.flatten_util import ravel_pytree

# TODO >> @danielkelshaw
#         Can we construct these based purely on a _step function?
#         Need to add type hints to these.
#         Make these work seamlessly with exp maps


def ravel_first_arg(f, unravel):
    @lu.transformation  # type: ignore
    def _generator(unravel, y_flat, *args):
        y = unravel(y_flat)
        ans = yield (y,) + args, {}
        ans_flat, _ = ravel_pytree(ans)
        yield ans_flat

    return _generator(lu.wrap_init(f), unravel).call_wrapped  # type: ignore


def _euler_step(f, dt, y, t, *args):
    return y + dt * f(y, t, *args)


def euler(f, dt, y0, t0, t1, *args):
    y0, unravel = ravel_pytree(y0)
    f = ravel_first_arg(f, unravel)
    out = _euler(f, dt, y0, t0, t1, *args)
    return unravel(out)


@ft.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _euler(f, dt, y0, t0, t1, *args):
    def fn_cond(carry):
        _, curr_t = carry
        return curr_t < t1

    def fn_body(carry):
        curr_y, curr_t = carry
        return (_euler_step(f, dt, curr_y, curr_t, *args), curr_t + dt)

    y1, t1 = jax.lax.while_loop(fn_cond, fn_body, (y0, t0))
    return y1


def _euler_fwd(f, dt, y0, t0, t1, *args):
    y1 = _euler(f, dt, y0, t0, t1, *args)
    return y1, (y1, t0, t1, args)


def _euler_rev(f, dt, res, g):
    y1, t0, t1, args = res

    def fn_augmented_dynamics(state, t, *args):
        curr_y, curr_a, *_ = state

        # note: we use -t here as we need to undo the negation
        y_dot, fn_vjp = jax.vjp(f, curr_y, -t, *args)
        return (-y_dot, *fn_vjp(curr_a))

    t1_bar = jnp.dot(f(y1, t1, *args), g)
    initial_state = (y1, g, t1_bar, jtu.tree_map(jnp.zeros_like, args))

    _, y_bar, t0_bar, args_bar = euler(fn_augmented_dynamics, dt, initial_state, -t1, -t0, *args)
    return (y_bar, t0_bar, t1_bar, *args_bar)


_euler.defvjp(_euler_fwd, _euler_rev)


def _rk4_step(f, dt, y, t, *args):
    k1 = f(y, t, *args)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt, *args)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt, *args)
    k4 = f(y + dt * k3, t + dt, *args)

    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rk4(f, dt, y0, t0, t1, *args):
    y0, unravel = ravel_pytree(y0)
    f = ravel_first_arg(f, unravel)
    out = _rk4(f, dt, y0, t0, t1, *args)
    return unravel(out)


@ft.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _rk4(f, dt, y0, t0, t1, *args):
    def fn_cond(carry):
        _, curr_t = carry
        return curr_t < t1

    def fn_body(carry):
        curr_y, curr_t = carry
        return (_rk4_step(f, dt, curr_y, curr_t, *args), curr_t + dt)

    y1, t1 = jax.lax.while_loop(fn_cond, fn_body, (y0, t0))
    return y1


def _rk4_fwd(f, dt, y0, t0, t1, *args):
    y1 = _rk4(f, dt, y0, t0, t1, *args)
    return y1, (y1, t0, t1, args)


def _rk4_rev(f, dt, res, g):
    y1, t0, t1, args = res

    def fn_augmented_dynamics(state, t, *args):
        curr_y, curr_a, *_ = state

        # note: we use -t here as we need to undo the negation
        y_dot, fn_vjp = jax.vjp(f, curr_y, -t, *args)
        return (-y_dot, *fn_vjp(curr_a))

    t1_bar = jnp.dot(f(y1, t1, *args), g)
    initial_state = (y1, g, t1_bar, jtu.tree_map(jnp.zeros_like, args))

    _, y_bar, t0_bar, args_bar = rk4(fn_augmented_dynamics, dt, initial_state, -t1, -t0, *args)
    return (y_bar, t0_bar, t1_bar, *args_bar)


_rk4.defvjp(_rk4_fwd, _rk4_rev)
