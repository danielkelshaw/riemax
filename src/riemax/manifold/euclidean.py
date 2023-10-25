import jax
import jax.numpy as jnp


def metric_tensor(x: jax.Array) -> jax.Array:
    r"""Defines the metric tensor for Euclidean space.

    In Euclidean space, the metric tensor is defined as the identity matrix

    $$
    g_{ij} = \delta_{ij}.
    $$

    !!! warning

        The Euclidean metric defined in this manner is not differentiable. This could cause problems in some places.

    Parameters:
        x: position $p \in \mathbb{R}$ at which to evaluate the metric tensor

    Returns:
        metric tensor in Euclidean space -- the identity matrix
    """

    return jnp.eye(N=x.shape[-1])


def distance(p: jax.Array, q: jax.Array) -> jax.Array:
    r"""Compute Euclidean distance between points.

    The Euclidean distance is simply defined by the L2 norm:

    $$
    d_E(p, q) = \lVert p - q \rVert_2.
    $$

    Parameters:
        p: position $p \in \mathbb{R}$ of the first point
        q: position $q \in \mathbb{R}$ of the second point

    Returns:
        euclidean distance between $p, q$
    """

    return jnp.sqrt(jnp.einsum('...i -> ...', (p - q) ** 2))
