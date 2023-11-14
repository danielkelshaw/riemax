import typing as tp

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from ._initializers import glorot_normal


class RWFParameters(tp.NamedTuple):
    mu: float = 1.0
    sigma: float = 0.1


class _RWFInitializer:
    def __init__(self, mu: float, sigma: float, w_init: hk.initializers.Initializer) -> None:
        self.mu = mu
        self.sigma = sigma
        self.w_init = w_init

    def __call__(self, shape: tp.Sequence[int], dtype: tp.Any) -> jax.Array:
        if not len(shape) == 2:
            raise ValueError('Must pass a shape of length two.')

        s1, s2 = shape

        w = self.w_init((s1 - 1, s2), dtype)
        g = jnp.exp(self.mu + hk.initializers.RandomNormal(self.sigma)((s2,), dtype))

        return jnp.concatenate((jnp.expand_dims(g, 0), w / g), 0)


class RandomWeightFactorization(hk.Module):
    def __init__(
        self,
        output_size: int,
        rwf_params: RWFParameters,
        with_bias: bool,
        w_init: hk.initializers.Initializer | None = None,
        b_init: hk.initializers.Initializer | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        self.output_size = output_size
        self.with_bias = with_bias

        if not (_w_init := w_init):
            _w_init = glorot_normal

        if not (_b_init := b_init):
            _b_init = hk.initializers.Constant(0.0)

        self.w_init = _RWFInitializer(*rwf_params, _w_init)
        self.b_init = _b_init

    def __call__(self, inputs: jax.Array, *, precision: jax.lax.Precision | None = None) -> jax.Array:
        if not inputs.shape:
            raise ValueError('Input must not be scalar.')

        input_size = self.input_size = inputs.shape[-1]

        w_init = self.w_init
        if not w_init:
            sigma = 1.0 / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=sigma)

        s, *v = hk.get_parameter('sv', shape=[input_size + 1, self.output_size], init=w_init)
        v = jnp.stack(v, 0)

        kernel = s * v
        out = jnp.dot(inputs, kernel, precision=precision)

        if self.with_bias:
            bias = hk.get_parameter(
                'bias',
                shape=[
                    self.output_size,
                ],
                init=self.b_init,
            )
            bias = jnp.broadcast_to(bias, out.shape)

            out += bias

        return out
