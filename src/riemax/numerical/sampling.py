import typing as tp

import jax
import jax.numpy as jnp


def _rwmh_kernel(
    key: jax.Array,
    fn: tp.Callable[[jax.Array], float],
    position: jax.Array,
    log_prob: jax.Array,
    sigma: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Conduct single update of Random-Walk Metropolis Hastings sampling.

    Parameters:
        key: random key to use for the sampling
        fn: function used to determine validity of samples
        position: position at which to continue the random walk
        log_prob: log probability of the initial position
        sigma: scaling parameter for the step size

    Returns:
        position: updated position in the sampling chain
        log_prob: the associated log probability at the updated position
        do_accept: whether to accept the sample or not
    """

    k1, k2 = jax.random.split(key)

    dpos_dt = jax.random.normal(k1, shape=position.shape) * sigma
    proposal = position + sigma * dpos_dt

    proposal_log_prob = jnp.log(fn(proposal))
    log_uniform = jnp.log(jax.random.uniform(k2))

    do_accept = log_uniform < (proposal_log_prob - log_prob)

    position = jnp.where(do_accept, proposal, position)
    log_prob = jnp.where(do_accept, proposal_log_prob, log_prob)

    return position, log_prob, do_accept


def rwmh_sampler(
    key: jax.Array,
    n_samples: int,
    fn: tp.Callable[[jax.Array], float],
    initial_position: jax.Array,
    burnin_steps: int = 20_000,
) -> tuple[jax.Array, jax.Array]:
    """Conduct Random-Walk Metropolis Hastings sampling.

    Parameters:
        key: random key to use for the sampling procedure
        n_samples: number of steps to conduct for the metropolis hastings sampling
        fn: function used to determine validity of samples
        initial_position: position at which to commence the random walk
        burnin_steps: number of initial steps to discard in the MCMC chain

    Returns:
        pos: positions of points sampled
        do_accept: whether or not to accept the sampled points
    """

    def mh_update(state, _):
        key, pos, log_prob, do_accept = state

        key, _ = jax.random.split(key)
        new_position, new_log_prob, do_accept = _rwmh_kernel(key, fn, pos, log_prob)

        return (key, new_position, new_log_prob, do_accept), state

    initial_state = (key, initial_position, jnp.log(fn(initial_position)), True)
    burnin_state, _ = jax.lax.scan(mh_update, initial_state, None, burnin_steps)

    _, (_, pos, _, do_accept) = jax.lax.scan(mh_update, burnin_state, None, n_samples)

    return pos, do_accept
