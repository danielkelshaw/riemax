import jax
import jax.numpy as jnp


def fn_peaks(x: jax.Array) -> jax.Array:
    """MATLAB peaks function.

    Parameters:
        x: position on the manifold, two-dimensions

    Returns:
        position on the surface, three-dimensions
    """

    a = 3 * (1 - x[0]) ** 2 * jnp.exp(-(x[0] ** 2) - (x[1] + 1) ** 2)
    b = 10 * (0.2 * x[0] - x[0] ** 3 - x[1] ** 5) * jnp.exp(-(x[0] ** 2) - x[1] ** 2)
    c = (1 / 3) * jnp.exp(-((x[0] + 1) ** 2) - x[1] ** 2)

    z = a - b - c

    return jnp.array([x[0], x[1], z])
