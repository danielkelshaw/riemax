import typing as tp

import jax
import jax.numpy as jnp


def construct_hypersphere(n: int) -> tp.Callable[[jax.Array], jax.Array]:
    def hypersphere(x: jax.Array) -> jax.Array:
        cos_x = jnp.concatenate((jnp.cos(x), jnp.array([1.0])))
        sin_x = jnp.concatenate((jnp.array([1.0]), jnp.sin(x)))

        return jnp.cumprod(sin_x) * cos_x

    return hypersphere


def euc_sphere_distance(p: jax.Array, q: jax.Array) -> jax.Array:
    return jnp.arccos(jnp.clip(jnp.einsum('i, i -> ', p, q), -1.0, 1.0))
