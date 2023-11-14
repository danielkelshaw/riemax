import typing as tp

import haiku as hk
import jax
import jax.numpy as jnp


class FourierEmbedding(hk.Module):
    def __init__(self, embedding_scale: float, embedding_dim: int, name: str | None = None) -> None:
        """Fourier embedding layer.

        Provides the ability to enforce periodicity along a given axis[^1]. This can be applied to arbitrary dynamical
        systems, even when the solution is not periodic -- the initial period can be set to greater than the length of
        the temporal domain.

        [^1]: Wang, Sifan, Shyam Sankaran, Hanwen Wang, and Paris Perdikaris. ‘An Expert’s Guide to Training Physics-Informed Neural Networks’. arXiv, 16 August 2023. http://arxiv.org/abs/2308.08468.

        """

        super().__init__(name=name)

        self.embed_scale = embedding_scale
        self.embed_dim = embedding_dim

        self.w_init = hk.initializers.RandomNormal(embedding_scale)

    def __call__(self, inputs: jax.Array, *, precision: jax.lax.Precision | None = None) -> jax.Array:
        kernel = hk.get_parameter('kernel', [inputs.shape[-1], self.embed_dim // 2], init=self.w_init)

        out = jnp.concatenate(
            [
                jnp.cos(jnp.dot(inputs, kernel, precision=precision)),
                jnp.sin(jnp.dot(inputs, kernel, precision=precision)),
            ],
            axis=-1,
        )

        return out


class PeriodicEmbedding(hk.Module):
    def __init__(
        self, axes: tp.Sequence[int], periods: tp.Sequence[float], trainable: tp.Sequence[bool], name: str | None = None
    ) -> None:
        super().__init__(name=name)

        # check period, axis, and trainable are the same length:
        if not len(set(map(len, (periods, axes, trainable)))) == 1:
            raise ValueError('All sequences must be the same length.')

        self.axes = axes
        self.trainable = trainable

        period_params = {}
        for idx, is_trainable in enumerate(trainable):
            if is_trainable:
                period_params[f'period_{idx}'] = hk.get_parameter(
                    f'period_{idx}', (), init=hk.initializers.Constant(periods[idx])
                )
            else:
                period_params[f'period_{idx}'] = periods[idx]

        self.period_params = period_params

    def __call__(self, inputs: jax.Array) -> jax.Array:
        y = []
        for axis_idx in range(inputs.shape[-1]):
            # if there is an assigned period, produce periodic embedding
            if axis_idx in self.axes:
                period = self.period_params[f'period_{self.axes.index(axis_idx)}']
                y.extend([jnp.cos(period * inputs[..., axis_idx]), jnp.sin(period * inputs[..., axis_idx])])

            else:
                y.append(inputs[..., axis_idx])

        return jnp.array(y).T
