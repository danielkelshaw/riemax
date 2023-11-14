import haiku as hk
import jax

from .random_weight_factorization import RandomWeightFactorization, RWFParameters


class Linear(hk.Module):
    def __init__(
        self,
        output_size: int,
        rwf_params: RWFParameters | None,
        with_bias: bool = True,
        w_init: hk.initializers.Initializer | None = None,
        b_init: hk.initializers.Initializer | None = None,
        name: str | None = None,
    ) -> None:
        """Generalised Linear Module.

        Extension of the base hk.Linear module, with added options for random weight factorisation[^1]. This provides an
        alternative means by which to parameterise a linear layer. This has been shown to provide improved training for
        continuous neural representations.

        [^1]: Wang, Sifan, Hanwen Wang, Jacob H. Seidman, and Paris Perdikaris. ‘Random Weight Factorization Improves the Training of Continuous Neural Representations’. arXiv, 5 October 2022. http://arxiv.org/abs/2210.01274."""

        super().__init__(name=name)

        network_kwargs = dict(output_size=output_size, with_bias=with_bias, w_init=w_init, b_init=b_init)

        match rwf_params:
            case RWFParameters(_mu, _sigma):
                self.module = RandomWeightFactorization(**(network_kwargs | dict(rwf_params=rwf_params)))  # type: ignore
            case None:
                self.module = hk.Linear(**network_kwargs)  # type: ignore
            case _:
                raise ValueError('Passed an invalid instance of rwf_params.')

    def __call__(self, x: jax.Array, *, precision: jax.lax.Precision | None = None) -> jax.Array:
        return self.module(x, precision=precision)
