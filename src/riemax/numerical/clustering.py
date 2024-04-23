import typing as tp

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from .frechet import GBOParams, frechet_mean


def closest_centroid(
    fn_distance: tp.Callable[[jax.Array, jax.Array], jax.Array], centroids: jax.Array, points: jax.Array
) -> jax.Array:
    fn_pairwise_distance = jax.vmap(jax.vmap(fn_distance, in_axes=(0, None)), in_axes=(None, 0))
    pairwise_distances = jnp.squeeze(fn_pairwise_distance(centroids, points))

    return jnp.argmin(pairwise_distances, -1)


def k_nearest_neighbours(
    fn_distance: tp.Callable[[jax.Array, jax.Array], jax.Array],
    initial_centroids: jax.Array,
    points: jax.Array,
    optimiser_params: GBOParams,
):
    fn_frechet_mean = jtu.Partial(frechet_mean, fn_distance, optimiser_params=optimiser_params)
    fn_closest_centroid = jtu.Partial(closest_centroid, fn_distance)
    centroids = initial_centroids

    prev_assigned_centroids = jnp.full(shape=(points.shape[0]), fill_value=-1)

    i = 0
    while True:
        print(f'On Iteration {i}')

        # assignment step
        assigned_centroids = fn_closest_centroid(centroids, points)
        curr_is_prev = (prev_assigned_centroids == assigned_centroids).all()

        if curr_is_prev or i > 30:
            break

        prev_assigned_centroids = assigned_centroids

        # update step
        def compute_centroid(z: int) -> jax.Array:
            associated_points = points[assigned_centroids == z]
            curr_centroid = centroids[z]
            (_, (mu, _)), _ = fn_frechet_mean(associated_points, curr_centroid)

            return mu

        centroids = jnp.stack(list(map(compute_centroid, range(initial_centroids.shape[0]))))
        i += 1

    assigned_centroids = fn_closest_centroid(centroids, points)

    return centroids, assigned_centroids
