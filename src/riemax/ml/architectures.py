import dataclasses
import typing as tp

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from .modules.embeddings import FourierEmbedding, PeriodicEmbedding
from .modules.linear import Linear
from .modules.modified_linear import ModifiedLinear
from .modules.random_weight_factorization import RWFParameters


def _get_final_activation(
    activation: tp.Callable[[jax.Array], jax.Array], activate_final: bool = False
) -> tp.Callable[[jax.Array], jax.Array]:
    def final_layer(x: jax.Array) -> jax.Array:
        if activate_final:
            return activation(x)

        return x

    return final_layer


class MLP(hk.Module):
    def __init__(
        self,
        output_sizes: tp.Sequence[int],
        with_bias: bool = True,
        w_init: hk.initializers.Initializer | None = None,
        b_init: hk.initializers.Initializer | None = None,
        activation: tp.Callable[[jax.Array], jax.Array] = jax.nn.tanh,
        activate_final: bool = False,
        rwf_params: RWFParameters | None = None,
        periodic_embedding: dict[str, tp.Any] | None = None,
        fourier_embedding: dict[str, tp.Any] | None = None,
        name: str | None = None,
    ) -> None:
        """Define standard MLP.

        Parameters:
            output_sizes: Output size of each layer of the network.
            with_bias: Whether to use bias in each layer or not.
            w_init: Initialiser for the weights of the network.
            b_init: Initialiser for the biases of the network.
            activation: Activation function to use to induce nonlinearities.
            activate_final: Whether to use activation function on output layer.
            rwf_params: Optional parameters for Random Weight Factorisation.
            periodic_embedding: Optional dictionary with parameters for periodic embeddings.
            fourier_embedding: Optional dictionary with parameters for fourier embeddings.
            name: Name to assign to haiku module.
        """

        super().__init__(name=name)

        self.activation = activation
        self.final_activation = _get_final_activation(activation, activate_final)

        self.periodic_embedding = periodic_embedding
        self.fourier_embedding = fourier_embedding

        layers = []
        for idx, output_size in enumerate(output_sizes):
            layers.append(Linear(output_size, rwf_params, with_bias, w_init, b_init, name=f'linear_{idx}'))  # type: ignore

        self.layers = tuple(layers)

    def __call__(self, inputs: jax.Array, *, precision: jax.lax.Precision | None = None) -> jax.Array:
        # compute periodic embeddings if necessary
        if self.periodic_embedding:
            inputs = PeriodicEmbedding(**self.periodic_embedding)(inputs)  # type: ignore

        # compute fourier embeddings if necessary
        if self.fourier_embedding:
            inputs = FourierEmbedding(**self.fourier_embedding)(inputs)  # type: ignore

        # run through layers
        out = inputs
        *hidden_layers, final_layer = self.layers

        for layer in hidden_layers:
            out = self.activation(layer(out, precision=precision))

        return self.final_activation(final_layer(out, precision=precision))


class ModifiedMLP(hk.Module):
    def __init__(
        self,
        output_sizes: tp.Sequence[int],
        with_bias: bool = True,
        w_init: hk.initializers.Initializer | None = None,
        b_init: hk.initializers.Initializer | None = None,
        activation: tp.Callable[[jax.Array], jax.Array] = jax.nn.tanh,
        activate_final: bool = False,
        rwf_params: RWFParameters | None = None,
        periodic_embedding: dict[str, tp.Any] | None = None,
        fourier_embedding: dict[str, tp.Any] | None = None,
        name: str | None = None,
    ) -> None:
        """Define Modified MLP.

        Parameters:
            output_sizes: Output size of each layer of the network.
            with_bias: Whether to use bias in each layer or not.
            w_init: Initialiser for the weights of the network.
            b_init: Initialiser for the biases of the network.
            activation: Activation function to use to induce nonlinearities.
            activate_final: Whether to use activation function on output layer.
            rwf_params: Optional parameters for Random Weight Factorisation.
            periodic_embedding: Optional dictionary with parameters for periodic embeddings.
            fourier_embedding: Optional dictionary with parameters for fourier embeddings.
            name: Name to assign to haiku module.
        """

        super().__init__(name=name)

        # check all layer sizes are identical
        if not len(set(output_sizes[:-1])) == 1:
            raise ValueError('Layer sizes must be identical to allow for encoding.')

        self.activation = activation
        self.final_activation = _get_final_activation(activation, activate_final)

        # generate layers of network
        *hidden_sizes, final_size = output_sizes
        encoding_size = hidden_sizes[0]

        # define input embeddings
        self.periodic_embedding = periodic_embedding
        self.fourier_embedding = fourier_embedding

        # define input encoders
        self.primary_encoder = Linear(encoding_size, rwf_params, with_bias, w_init, b_init, name='PrimaryEncoder')  # type: ignore
        self.secondary_encoder = Linear(encoding_size, rwf_params, with_bias, w_init, b_init, name='SecondaryEncoder')  # type: ignore

        # define layers of the network
        layers = []
        for idx, output_size in enumerate(hidden_sizes):
            layers.append(
                ModifiedLinear(output_size, activation, rwf_params, with_bias, w_init, b_init, name=f'linear_{idx}')  # type: ignore
            )

        layers.append(Linear(final_size, rwf_params, with_bias, w_init, b_init, name=f'layer_{idx + 1}'))  # type: ignore

        # finalise layers of the network
        self.layers = tuple(layers)

    def __call__(self, inputs: jax.Array, *, precision: jax.lax.Precision | None = None) -> jax.Array:
        # compute periodic embeddings if necessary
        if self.periodic_embedding:
            inputs = PeriodicEmbedding(**self.periodic_embedding)(inputs)  # type: ignore

        # compute fourier embeddings if necessary
        if self.fourier_embedding:
            inputs = FourierEmbedding(**self.fourier_embedding)(inputs)  # type: ignore

        # compute input encodings
        u = self.activation(self.primary_encoder(inputs, precision=precision))
        v = self.activation(self.secondary_encoder(inputs, precision=precision))

        # run through layers
        out = inputs
        *hidden_layers, final_layer = self.layers

        for layer in hidden_layers:
            out = layer(out, u, v, precision=precision)

        return self.final_activation(final_layer(out, precision=precision))


@dataclasses.dataclass
class Encoder(hk.Module):
    output_sizes: tp.Sequence[int]
    latent_size: int

    activation: tp.Callable[[jax.Array], jax.Array]

    def __call__(self, x: jax.Array) -> jax.Array:
        x = hk.Flatten()(x)  # type: ignore

        for s in self.output_sizes:
            x = self.activation(hk.Linear(s)(x))  # type: ignore

        return self.activation(hk.Linear(self.latent_size)(x))  # type: ignore


@dataclasses.dataclass
class Decoder(hk.Module):
    output_sizes: tp.Sequence[int]
    output_shape: tp.Sequence[int]

    activation: tp.Callable[[jax.Array], jax.Array]

    def __call__(self, x: jax.Array) -> jax.Array:
        for s in self.output_sizes:
            x = self.activation(hk.Linear(s)(x))  # type: ignore

        x = hk.Linear(np.prod(self.output_shape))(x)  # type: ignore

        return jnp.reshape(x, (-1, *self.output_shape))


@dataclasses.dataclass
class Autoencoder(hk.Module):
    encoder: Encoder
    decoder: Decoder

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.astype(jnp.float32)
        return self.decoder(self.encoder(x))


def construct_autoencoder(
    output_sizes: tp.Sequence[int],
    latent_size: int,
    output_shape: tp.Sequence[int],
    activation: tp.Callable[[jax.Array], jax.Array] = jax.nn.tanh,
) -> hk.MultiTransformed:
    def _model():
        encoder = Encoder(output_sizes=output_sizes, latent_size=latent_size, activation=activation)  # type: ignore
        decoder = Decoder(output_sizes=list(reversed(output_sizes)), output_shape=output_shape, activation=activation)  # type: ignore

        autoencoder = Autoencoder(encoder, decoder)  # type: ignore

        def __init(x: jax.Array) -> jax.Array:
            return autoencoder(x)

        return __init, (encoder, decoder, autoencoder)

    return hk.without_apply_rng(hk.multi_transform(_model))
