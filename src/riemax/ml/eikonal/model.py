from __future__ import annotations

import typing as tp

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ...manifold.euclidean import distance as euclidean_distance
from ...manifold.geometry import inner_product, metric_tensor, pullback
from ...manifold.types import M


def standardise(x: jax.Array, lower: jax.Array, upper: jax.Array) -> jax.Array:
    return 2.0 * (x - lower) / (upper - lower) - 1.0


type EikonalNetwork = tp.Callable[[hk.Params, jax.Array, jax.Array], jax.Array]


def _maybe_standardise(
    fn_standardise: tp.Callable[[jax.Array], jax.Array] | None,
) -> tp.Callable[[jax.Array], jax.Array]:
    def _inner(x: jax.Array) -> jax.Array:
        if fn_standardise:
            return fn_standardise(x)

        return x

    return _inner


class BaseConstrainedPhiModel(hk.Module):
    def __init__(
        self,
        module: hk.Module,
        fn_transformation: tp.Callable[[jax.Array], jax.Array],
        fn_standardise: tp.Callable[[jax.Array], jax.Array],
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        self.phi = module

        self.fn_transformation = fn_transformation

        _metric = jtu.Partial(metric_tensor, fn_transformation=fn_transformation)
        self.fn_inner_product = jtu.Partial(inner_product, metric=_metric)

        self.fn_standardise = _maybe_standardise(fn_standardise)
        self.fn_distance = pullback(euclidean_distance, fn_transformation)

    def _upper_bound(self, p: M[jax.Array], q: M[jax.Array]) -> jax.Array:
        t = jnp.linspace(0.0, 1.0, 100 + 1)

        gamma = p + jnp.einsum('i, j -> ij', t, (q - p))
        dpq = (q - p) / 100

        piecewise_distance = jnp.sqrt(
            hk.vmap(self.fn_inner_product, in_axes=(0, None, None), split_rng=False)(gamma, dpq, dpq)  # type: ignore
        )

        return jnp.sum(piecewise_distance)

    def __call__(self, p: M[jax.Array], q: M[jax.Array]) -> jax.Array:
        raise NotImplementedError('Need to overload this class.')


class OnewayEikonal(BaseConstrainedPhiModel):
    def __init__(
        self,
        module: hk.Module,
        fn_transformation: tp.Callable[[jax.Array], jax.Array],
        fn_standardise: tp.Callable[[jax.Array], jax.Array],
        name: str | None = None,
    ) -> None:
        super().__init__(module, fn_transformation, fn_standardise, name)

    def __call__(self, p: M[jax.Array], q: M[jax.Array]) -> jax.Array:
        standardised_q = self.fn_standardise(q)
        return self.fn_distance(p, q) * (1.0 + jax.nn.softplus(self.phi(standardised_q)))


class TwowayEikonal(BaseConstrainedPhiModel):
    def __init__(
        self,
        module: hk.Module,
        fn_transformation: tp.Callable[[jax.Array], jax.Array],
        fn_standardise: tp.Callable[[jax.Array], jax.Array],
        name: str | None = None,
    ) -> None:
        super().__init__(module, fn_transformation, fn_standardise, name)

    def __call__(self, p: M[jax.Array], q: M[jax.Array]) -> jax.Array:
        standardised_p, standardised_q = map(self.fn_standardise, (p, q))

        pq = jnp.concatenate((standardised_p, standardised_q))
        qp = jnp.concatenate((standardised_q, standardised_p))

        mean_output = 0.5 * (self.phi(pq) + self.phi(qp))
        return self.fn_distance(p, q) * (1.0 + jax.nn.softplus(mean_output))


class OnewayEikonalUB(BaseConstrainedPhiModel):
    def __init__(
        self,
        module: hk.Module,
        fn_transformation: tp.Callable[[jax.Array], jax.Array],
        fn_standardise: tp.Callable[[jax.Array], jax.Array],
        name: str | None = None,
    ) -> None:
        super().__init__(module, fn_transformation, fn_standardise, name)

    def __call__(self, p: M[jax.Array], q: M[jax.Array]) -> jax.Array:
        standardised_q = self.fn_standardise(q)

        lower_bound = self.fn_distance(p, q)
        upper_bound = self._upper_bound(p, q)

        return (upper_bound - lower_bound) * jax.nn.sigmoid(self.phi(standardised_q)) + lower_bound


class TwowayEikonalUB(BaseConstrainedPhiModel):
    def __init__(
        self,
        module: hk.Module,
        fn_transformation: tp.Callable[[jax.Array], jax.Array],
        fn_standardise: tp.Callable[[jax.Array], jax.Array],
        name: str | None = None,
    ) -> None:
        super().__init__(module, fn_transformation, fn_standardise, name)

    def __call__(self, p: M[jax.Array], q: M[jax.Array]) -> jax.Array:
        standardised_p, standardised_q = map(self.fn_standardise, (p, q))

        pq = jnp.concatenate((standardised_p, standardised_q))
        qp = jnp.concatenate((standardised_q, standardised_p))

        lower_bound = self.fn_distance(p, q)
        upper_bound = self._upper_bound(p, q)

        mean_output = 0.5 * (self.phi(pq) + self.phi(qp))
        return (upper_bound - lower_bound) * jax.nn.sigmoid(mean_output) + lower_bound
