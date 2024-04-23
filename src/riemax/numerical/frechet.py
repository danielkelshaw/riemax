import typing as tp

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from ..manifold.types import M


class FrechetState(tp.NamedTuple):
    mu: M[jax.Array]
    opt_state: optax.OptState


def _frechet_loss(
    fn_distance: tp.Callable[[jax.Array, jax.Array], jax.Array], mu: M[jax.Array], q: M[jax.Array]
) -> jax.Array:
    """Compute Frechet loss.

    Parameters:
        fn_distance: Function used to compute distance on the manifold.
        mu: Current esimate of the Frechet mean.
        q: Samples on the manifold.

    Returns:
        Frechet loss.
    """

    d_mu_q = jax.vmap(fn_distance, in_axes=(None, 0))(mu, q)
    return jnp.mean(jnp.square(d_mu_q))


class GBOParams(tp.NamedTuple):
    optimiser: optax.GradientTransformation
    n_updates: int


def frechet_mean(
    fn_distance: tp.Callable[[jax.Array, jax.Array], jax.Array],
    points: jax.Array,
    initial_guess: jax.Array,
    optimiser_params: GBOParams,
) -> tuple[tuple[jax.Array, FrechetState], tuple[jax.Array, FrechetState]]:
    """Compute Frechet mean

    Parameters:
        fn_distance: Function used to compute distance on the manifold.
        points: Samples on the manifold.
        initial_guess: Initial guess of the mean on the manifold
        optimiser_params: Parameters for gradient-based optimisation

    Returns:
        final: Final result of the Frechet mean
        preceding: Optimisation trajectory.
    """

    opt_state = optimiser_params.optimiser.init(initial_guess)
    initial_state = FrechetState(mu=initial_guess, opt_state=opt_state)

    fn_frechet_loss = jtu.Partial(_frechet_loss, fn_distance)

    # can we use `construct_update` here?
    def frechet_update(state: FrechetState) -> tuple[jax.Array, FrechetState]:
        loss, grads = jax.value_and_grad(fn_frechet_loss, argnums=0)(state.mu, points)

        updates, opt_state = optimiser_params.optimiser.update(grads, state.opt_state, params=state.mu)
        updated_mu = tp.cast(jax.Array, optax.apply_updates(state.mu, updates))

        return loss, FrechetState(mu=updated_mu, opt_state=opt_state)

    def _body(
        carry: tuple[jax.Array, FrechetState], _: None
    ) -> tuple[tuple[jax.Array, FrechetState], tuple[jax.Array, FrechetState]]:
        loss, state = carry
        next_carry = frechet_update(state)

        return next_carry, carry

    return jax.lax.scan(_body, (jnp.array(jnp.inf), initial_state), None, length=optimiser_params.n_updates)
