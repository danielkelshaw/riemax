import functools as ft
import typing as tp

import jax
import jax.numpy as jnp

from ._marked import manifold_marker
from .types import M, MetricFn


@manifold_marker.mark(jittable=False)
def grad(fn: tp.Callable[[M[jax.Array]], jax.Array], metric: MetricFn) -> tp.Callable[[M[jax.Array]], jax.Array]:

    r"""Compute gradient of scalar function on the manifold.

    When an inner product $\langle \cdot, \cdot \rangle$ is defined, the gradient $\nabla f$ of a function $f$ is
    defined as the unique vector $V$ such that its inner product with any element of $V$ is the directional derivative
    of $f$ along the vector.[^1][^2] Precisely

    $$
    \langle \nabla f, \cdot \rangle = df = \partial_i f dx^i,
    $$

    this yields

    $$
    \nabla f = (df)^\sharp = g^{ij} \partial_j f
    $$

    The 1-form $df$ is a section of the cotangent bundle, giving a local linear approximation to $f$ in the cotangent
    space at each point.

    **Example:**

    Given a scalar function $f$, we can define the gradient as

    ```python
    # ...

    def scalar_fn(p: jax.Array) -> jax.Array:
        return jnp.sum(jnp.square(p))

    fn_grad = riemax.manifold.operators.grad(scalar_fn, fn_metric)
    ```

    [^1]: Carmo, Manfredo Perdigão do. Riemannian Geometry. 2013.
    [^2]: Lee, John M. Introduction to Riemannian Manifolds. 2018.

    Parameters:
        fn: scalar function to take derivative of
        metric: function defining the metric tensor on the manifold

    Returns:
        transformed: function which takes gradient of scalar function on the manifold
    """

    @ft.wraps(fn)
    def transformed(x: M[jax.Array]) -> jax.Array:

        co_gx = metric(x)
        contra_gx = jnp.linalg.inv(co_gx)

        fn_j = jax.jacfwd(fn)(x)

        return jnp.einsum('ij, ...j -> i', contra_gx, fn_j)

    return transformed


@manifold_marker.mark(jittable=False)
def div(fn: tp.Callable[[M[jax.Array]], jax.Array], metric: MetricFn) -> tp.Callable[[M[jax.Array]], jax.Array]:

    r"""Compute divergence of vector-valued function on the manifold.

    Given a vector field $X \in TM$, we define the divergence as[^1][^2]

    $$
    \nabla \cdot X = \lvert g \rvert^{-\frac{1}{2}} \partial_i \left( \lvert g \rvert^{\frac{1}{2}} X^i \right)
    $$

    [^1]: Carmo, Manfredo Perdigão do. Riemannian Geometry. 2013.
    [^2]: Lee, John M. Introduction to Riemannian Manifolds. 2018.

    Parameters:
        fn: vector function to compute divergence of
        metric: function defining the metric tensor on the manifold

    Returns:
        transformed: Function which computes divergence of vector-valued function on the manifold
    """

    @ft.wraps(fn)
    def transformed(x: M[jax.Array]) -> jax.Array:

        co_gx = metric(x)
        sqrt_det_co_gx = jnp.sqrt(jnp.linalg.det(co_gx))

        def _inner(_x: M[jax.Array]) -> jax.Array:

            co_gx_inner = metric(_x)
            sqrt_det_co_gx_inner = jnp.sqrt(jnp.linalg.det(co_gx_inner))

            val = sqrt_det_co_gx_inner * fn(_x)

            if val.ndim == 0:
                raise ValueError('div only defined for vector fields.')

            return val

        inner_i = jax.jacfwd(_inner)(x)

        return jnp.einsum('ii -> ', inner_i) / sqrt_det_co_gx

    return transformed


@manifold_marker.mark(jittable=False)
def laplace_beltrami(fn: tp.Callable[[M[jax.Array]], jax.Array], metric: MetricFn) -> tp.Callable[[M[jax.Array]], jax.Array]:

    r"""Compute laplacian of scalar-valued function on the manifold.

    Given a function $f: M \rightarrow \mathbb{R}$, we can compute the Laplacian by taking the divergence of the
    exterior derivative.[^1][^2] Precisely, we can compute

    $$
    \Delta X = \lvert g \rvert^{-\frac{1}{2}} \partial_i \left( \lvert g \rvert^{\frac{1}{2}} g^{ij} \partial_j f \right)
    $$

    [^1]: Carmo, Manfredo Perdigão do. Riemannian Geometry. 2013.
    [^2]: Lee, John M. Introduction to Riemannian Manifolds. 2018.

    Parameters:
        fn: scalar function to compute laplacian of
        metric: function defining the metric tensor on the manifold

    Returns:
        transformed: Function which computes laplacian of scalar-valued functions on the manifold.
    """

    @ft.wraps(fn)
    def transformed(x: M[jax.Array]) -> jax.Array:
        return div(grad(fn, metric), metric)(x)

    return transformed
