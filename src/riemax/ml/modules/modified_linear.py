import typing as tp

import haiku as hk
import jax

from .linear import Linear
from .random_weight_factorization import RWFParameters


class ModifiedLinear(hk.Module):
    def __init__(
        self,
        output_size: int,
        activation: tp.Callable[[jax.Array], jax.Array],
        rwf_params: RWFParameters | None,
        with_bias: bool = True,
        w_init: hk.initializers.Initializer | None = None,
        b_init: hk.initializers.Initializer | None = None,
        name: str | None = None,
    ) -> None:
        """Modified Linear Layer.

        Provides an implementation of the modified linear layer[^1]. This layer allows us to provide embedding
        information which shows improved performance when training physics-informed neural networks.

        [^1]: Wang, Sifan, Shyam Sankaran, Hanwen Wang, and Paris Perdikaris. ‘An Expert’s Guide to Training Physics-Informed Neural Networks’. arXiv, 16 August 2023. http://arxiv.org/abs/2308.08468.
        """

        super().__init__(name=name)

        self.module = Linear(output_size, rwf_params, with_bias, w_init, b_init, name=None)  # type: ignore
        self.activation = activation

    def __call__(
        self,
        x: jax.Array,
        primary_encoding: jax.Array,
        secondary_encoding: jax.Array,
        *,
        precision: jax.lax.Precision | None = None,
    ) -> jax.Array:
        sigma_f = self.activation(self.module(x, precision=precision))
        return sigma_f * primary_encoding + (1.0 - sigma_f) * secondary_encoding
