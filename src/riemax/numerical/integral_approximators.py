import typing as tp

import jax
import jax.numpy as jnp

type IntegralApproximationFn = tp.Callable[[jax.Array, float], jax.Array]


def mean_integration(fx: jax.Array, _: float) -> jax.Array:
    """Integration by taking mean.

    Parameters:
        fx: vector of function evaluations to integrate
        h: sample spacing

    Returns:
        integrated value
    """

    if fx.ndim != 1:
        raise ValueError('Must pass a 1D array to integrate.')

    return jnp.mean(fx)


def trapezoidal_integration(fx: jax.Array, h: float) -> jax.Array:
    """Integration by trapezoidal rule.

    Parameters:
        fx: vector of function evaluations to integrate
        h: sample spacing

    Returns:
        integrated value
    """

    if fx.ndim != 1:
        raise ValueError('Must pass a 1D array to integrate.')

    return (h / 2.0) * (fx[0] + fx[-1] + 2.0 * jnp.sum(fx[1:-1]))


def simpsons_integration(fx: jax.Array, h: float) -> jax.Array:
    """Integration by Simpsons 1/3 rule.

    Parameters:
        fx: vector of function evaluations to integrate
        h: sample spacing

    Returns:
        integrated value
    """

    if fx.ndim != 1:
        raise ValueError('Must pass a 1D array to integrate()')

    return (h / 3.0) * (fx[0] + fx[-1] + 4.0 * jnp.sum(fx[1:-1:2]) + 2.0 * jnp.sum(fx[2:-1:2]))
