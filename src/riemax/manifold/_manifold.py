from __future__ import annotations

import typing as tp

import jax
import jax.tree_util as jtu

from ._marked import manifold_marker
from .geometry import metric_tensor
from .types import M, MetricFn


class Manifold:
    """Convenience class for creating a manifold.

    !!! note

        Riemax remains a functional library, and all functionality has been defined in a pure manner. The `Manifold`
        class ties together relevant functionlity, automatically creating partial applications of functions so the user
        need not pass the metric function around all the time.

        This class is defined dynamically, using decorators to mark relevant functions throughout the library.

    ### Instantiation

    A default `__init__` is provided, allowing the user to pass a function which computes the metric tensor on the given
    manifold. Alternatively, you can instantiate an `Manifold` using a function transformation:

    ```python
    manifold = riemax.Manifold.from_fn_transformation(...)
    ```

    ### Methods

    After you have defined your manifold, you have the freedom to compute quantities without passing in the metric
    tensor each time. Most of the time, you will be working on a single manifold, and creating partial functions can
    become tiresome. Instead, the `Manifold` class handles the partial applications, as well as `jax.jit`, letting you
    call functions naturally.

    For example you can compute geometric quantities:

    ```python
    metric_p = manifold.metric_tensor(p)
    riemann_tensor = manifold.sk_riemann_tensor(p)
    ```
    """

    def __new__(cls, metric: MetricFn, *, jit: bool = True) -> Manifold:
        obj = super().__new__(cls)
        for fn, jittable in manifold_marker:
            p_fn = jtu.Partial(fn, metric=metric)

            if jittable and jit:
                p_fn = jax.jit(p_fn)

            p_fn.__doc__ = fn.__doc__

            setattr(obj, fn.__name__, p_fn)

        return obj

    def __init__(self, metric: MetricFn, *, jit: bool = True) -> None:
        """Instantiate a Manifold

        Parameters:
            metric: function defining the metric tensor on the manifold.
            jit: whether to apply `jax.jit` to the functions.
        """

        self.metric_tensor = metric
        self.jit = jit

    def __repr__(self) -> str:
        return f'Manifold(metric={self.metric_tensor.__name__})'

    @classmethod
    def from_fn_transformation(
        cls, fn_transformation: tp.Callable[[M[jax.Array]], jax.Array], *, jit: bool = True
    ) -> Manifold:
        """Instantiates an induced Manifold from a function transformation.

        Parameters:
            fn_transformation: function used to define the induced metric.
            jit: whether to apply `jax.jit` to the functions.
        """

        p_metric = jtu.Partial(metric_tensor, fn_transformation=fn_transformation)
        p_metric.__doc__ = metric_tensor.__doc__

        return cls(metric=p_metric, jit=jit)
