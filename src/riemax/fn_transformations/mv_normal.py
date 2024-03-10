import typing as tp

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def produce_mv_normal_manifold(
    alpha: jax.Array, mu: jax.Array, cov: jax.Array
) -> tuple[
    tp.Callable[[jax.Array], jax.Array],
    tp.Callable[[jax.Array, tuple[int, ...]], jax.Array],
    tuple[jax.Array, jax.Array],
]:
    """Produce multivariate normal manifold and sampler.

    Parameters:
        alpha: Scaling parameters for the distributions.
        mu: Array of mean locations for the distributions.
        cov: Covariance matrices for each of the distributions.

    Returns:
        fn_manifold: Function which maps coordinate to pdf realisation.
        fn_manifold_sampler: Function to sample on the resulting manifold.
        mv_mu: Known mean on the manifold.
    """

    fn_pdf = jax.vmap(jsp.stats.multivariate_normal.pdf, in_axes=(None, 0, 0))

    def fn_manifold(x: jax.Array) -> jax.Array:
        y = jnp.expand_dims(jnp.einsum('i, i... -> ...', alpha, fn_pdf(x, mu, cov)), -1)
        return jnp.concatenate([x, y], -1)

    mv_mu = jnp.einsum('i, i... -> ...', alpha, mu)
    mv_cov = jnp.einsum('i, i... -> ...', alpha**2, cov)

    def fn_manifold_sampler(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        return jax.random.multivariate_normal(key, mv_mu, mv_cov, shape=shape)

    return fn_manifold, fn_manifold_sampler, (mv_mu, mv_cov)


def fn_manifold(alpha: jax.Array, mu: jax.Array, cov: jax.Array) -> tp.Callable[[jax.Array], jax.Array]:
    """Simply produce the manifold function.

    Parameters:
        alpha: Scaling parameters for the distributions.
        mu: Array of mean locations for the distributions.
        cov: Covariance matrices for each of the distributions.

    Returns:
        fn_manifold: Function which maps coordinate to pdf realisation.
    """

    fn, _, _ = produce_mv_normal_manifold(alpha, mu, cov)
    return fn
