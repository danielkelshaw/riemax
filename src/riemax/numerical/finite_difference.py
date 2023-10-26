import typing as tp

import jax
import jax.numpy as jnp


def _perturb(x: jax.Array, dh: float) -> jax.Array:
    """Provide the pertubation required for finite difference.

    Parameters:
        x: input value to perturb
        dh: perturbation to apply

    Returns:
        perturbed input values
    """

    ndim = x.shape[-1]

    def pswitch(v, branch, ps):
        return jax.lax.switch(branch, (jax.lax.add, jax.lax.sub), v, ps)

    branch_idx = jnp.arange(2)
    perturbations = dh * jnp.eye(ndim, ndim)

    fn = jax.vmap(jax.vmap(pswitch, in_axes=(None, 0, None)), in_axes=(None, None, 0))
    return fn(x, branch_idx, perturbations)


def central_difference(fn: tp.Callable[[jax.Array], jax.Array], x: jax.Array, dh: float = 1e-4) -> jax.Array:
    """Compute second-order central-difference for arbitrary PyTrees.

    Parameters:
        fn: function to compute the gradient of
        x: position to compute gradient at
        dh: pertubation required for the finite difference

    Returns:
        computed derivative of the function at x
    """

    xp = _perturb(x, dh)

    fn_eval = jax.vmap(jax.vmap(fn))(xp)
    return jax.vmap(lambda x: jax.lax.sub(*x))(fn_eval) / (2.0 * dh)
