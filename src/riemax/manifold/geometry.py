import typing as tp

import einops
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ._marked import manifold_marker
from .types import M, MetricFn, TpM


def pullback[**P, T](f: tp.Callable[P, T], fn_transformation: tp.Callable[[jax.Array], jax.Array]) -> tp.Callable[P, T]:
    r"""Define the pullback of a function by a transformation.

    Let $\iota: M \hookrightarrow N$ be a smooth immersion, and suppose $f: N \rightarrow S$ is a smooth function on N.
    Then we define the **pullback** of $f$ by $\iota$ as the smooth function $\iota^\ast f: M \rightarrow S$, or:

    $$
    (\iota^\ast f)(x) = f(\iota(x))
    $$

    **Example:**

    One notable example is computing the Euclidean distance $d_E(p, q) = \lVert p - q \rVert_2$ between two points on
    the manifold, we can use the pullback to do this.

    In code, we may write something like

    ```python
    # ...


    def euclidean_distance(p: jax.Array, q: jax.Array) -> jax.Array:
        return jnp.sum(jnp.square(p - q))


    # pullback_distance: Callable[[M[jax.Array]], Rn[jax.Array]]
    pullback_distance = riemax.geometry.pullback(euclidean_distance, fn_transformation)
    ```

    [^1]: Carmo, Manfredo Perdigão do. Differential Geometry of Curves & Surfaces. 2018.
    [^2]: Lee, John M. Introduction to Smooth Manifolds. 2012.

    Parameters:
        f: function, $f: N \rightarrow S$, to pullback.
        fn_transformation: function defining smooth immersion, $\iota: M \hookrightarrow N$

    Returns:
        f_pullback: pullback of f by fn_transformation, $\iota^\ast f: M \rightarrow S$
    """

    def f_pullback(*args: P.args, **kwargs: P.kwargs) -> T:
        args = jtu.tree_map(fn_transformation, args)
        kwargs = jtu.tree_map(fn_transformation, kwargs)

        return f(*args, **kwargs)

    return f_pullback


def metric_tensor(x: M[jax.Array], fn_transformation: tp.Callable[[M[jax.Array]], jax.Array]) -> jax.Array:
    r"""Computes the covariant metric tensor at a point $p \in M$

    Given a smooth immersion $\iota: M \hookrightarrow N$, we can define the induced metric:

    $$
    g_{ij} = \frac{\partial \iota}{\partial x^i}\frac{\partial \iota}{\partial x^j}
    $$

    For the given point $p \in M$, the induced metric $g$ allows us to operate locally on the tangent space $T_p M$.
    Precisely, the induced metric is a symmetric bilinear form $g: T_p M \times T_p M \rightarrow \mathbb{R}$. This
    allows us to compute distances and angles in the tangent space, and is used to compute most intrinsic quantities
    on the manifold.[^1][^2]

    [^1]: Carmo, Manfredo Perdigão do. Riemannian Geometry. 2013.
    [^2]: Lee, John M. Introduction to Riemannian Manifolds. 2018.

    Parameters:
        x: position $p \in M$ at which to evaluate the metric tensor
        fn_transformation: function defining smooth immersion, $\iota: M \hookrightarrow N$

    Returns:
        covariant metric tensor, $g_{ij}(p)$
    """

    fn_jacobian = jax.jacobian(fn_transformation)(x)
    return jnp.einsum('ki, kj -> ij', fn_jacobian, fn_jacobian)


@manifold_marker.mark(jittable=True)
def inner_product(x: M[jax.Array], v: TpM[jax.Array], w: TpM[jax.Array], metric: MetricFn) -> jax.Array:
    r"""Compute inner product on the tanget plane, $g_p (v, w)$.

    The inner product is essential for computing the magnitude of vectors on the tangent space and can be used in
    computing the angles between two tangent vectors.

    Parameters:
        x: position $p \in M$ at which to evaluate the inner product
        v: first vector on the tangent space, $v \in T_p M$
        w: second vector on the tangent space, $w \in T_p M$
        metric: function defining the metric tensor on the manifold

    Returns:
        inner product between $v, w \in T_p M$ computed at $p \in M$
    """

    return jnp.einsum('ij, i, j -> ', metric(x), v, w)


@manifold_marker.mark(jittable=True)
def magnitude(x: M[jax.Array], v: TpM[jax.Array], metric: MetricFn) -> jax.Array:
    r"""Compute length of vector on the tangent space, $\lVert v \rVert$

    The metric $g$ provides the ability to compute the inner product on the tangent space. Using the standard definition
    of the inner product, we can compute the length of a vector as

    $$
    \lVert v \rVert = \langle v, v \rangle^{0.5}_g = \sqrt{g_p(v, v)}
    $$

    Parameters:
        x: position $p \in M$ at which the tangent vector is defined
        v: tangent vector $v$ to compute the magnitude of
        metric: function defining the metric tensor on the manifold

    Returns:
        length of the tangent vector $v$ at the point $p \in M$
    """

    return jnp.sqrt(inner_product(x, v, v, metric))


@manifold_marker.mark(jittable=True)
def contravariant_metric_tensor(x: M[jax.Array], metric: MetricFn) -> jax.Array:
    r"""Computes inverse of the metric tensor.

    We observe that the identity $g_{ij} g^{ij} = I$ holds. This function allows us to compute the inverse of the
    covariant metric tensor, an important tool allowing us to raise indices

    $$
    v^i = g^{ij} v_j = (v_i)^\sharp
    $$

    !!! warning "Computing the Inverse:"

        At the moment, we explicitly take the inverse of the covariant metric tensor by using `jnp.linalg.inv`. For
        large systems, we may want to solve for this instead.

    Parameters:
        x: position $p \in M$ at which to evaluate the inverse of the metric tensor
        metric: function defining the metric tensor on the manifold

    Returns:
        contravariant metric tensor.
    """

    return jnp.linalg.inv(metric(x))


@manifold_marker.mark(jittable=True)
def fk_christoffel(x: M[jax.Array], metric: MetricFn) -> jax.Array:
    r"""Christoffel symbols of the first kind $\Gamma_{kij} = \left[ ij, k \right]$.

    These Christoffel symbols are components of the affine connection, defined as

    $$
    \Gamma_{kij} = 0.5 \left( \partial g_{ki, j} + \partial g_{kj, i} - \partial g_{ij, k} \right).
    $$

    These allow us to compute the geodesic equation, measures of curvature, and more.[^1][^2]

    [^1]: Carmo, Manfredo Perdigão do. Differential Geometry of Curves & Surfaces. 2018.
    [^2]: Lee, John M. Introduction to Smooth Manifolds. 2012.

    Parameters:
        x: position $p \in M$ at which to evaluate Christoffel symbols
        metric: function defining the metric tensor on the manifold

    Returns:
        christoffel symbols of the first kind.
    """

    dgdx = jax.jacobian(metric)(x)

    def get_value(k, i, j) -> jax.Array:
        return 0.5 * (dgdx[k, i, j] + dgdx[k, j, i] - dgdx[i, j, k])

    return jnp.vectorize(get_value)(*jnp.indices(dgdx.shape))


@manifold_marker.mark(jittable=True)
def sk_christoffel(x: M[jax.Array], metric: MetricFn) -> jax.Array:
    r"""Christoffel symbols of the second kind: $\Gamma^k_{\phantom{k}ij} = \left\{ ij, k \right\}$

    The Christoffel symbols of the second-kind are simply index-raised versions of the Christoffel symbols of the first
    kind. We simply use the inverse of the metric tensor to raise the index,

    $$
    \Gamma^k_{\phantom{k}ij} = g^{km} \Gamma_{mij}.
    $$

    Parameters:
        x: position $p \in M$ at which to evaluate Christoffel symbols of the second kind
        metric: function defining the metric tensor on the manifold

    Returns:
        Christoffel symbol of the second kind.
    """

    fk_christ = fk_christoffel(x, metric)
    contravariant_g_ij = contravariant_metric_tensor(x, metric)

    return jnp.einsum('kn, nij -> kij', contravariant_g_ij, fk_christ)


@manifold_marker.mark(jittable=True)
def sk_riemann_tensor(x: M[jax.Array], metric: MetricFn) -> jax.Array:
    r"""Compute Riemann curvature tensor of the second kind, $R^i_{\phantom{i}jkl}$

    The Riemann tensor provides a notion of curvature on the manifold. It is defined in terms of covariant derivatives

    $$
    R(X, Y) = \left[ \nabla_X, \nabla_Y \right] - \nabla_{\left[X, Y \right]}
    $$

    This is expressible entirely in terms of the Christoffel symbols, which are used in the computation.

    Parameters:
        x: position $p \in M$ at which to evaluate the Riemann curvature tensor.
        metric: function defining the metric tensor on the manifold

    Returns:
        Riemann curvature tensor of the second kind.
    """

    # create partial to allow us to compute Jacobian easily
    fn_sk_christoffel = jtu.Partial(sk_christoffel, metric=metric)
    p_fn_sk_christoffel = jax.jacfwd(fn_sk_christoffel)

    # christoffel symbols [ij, k]
    g_ijk = fn_sk_christoffel(x)

    # compute the second term, [ij, k]_{,m}
    g_ijk_m = p_fn_sk_christoffel(x)

    # compute first term via transposition of the first term
    g_ijm_k = einops.rearrange(g_ijk_m, 'i j k m -> i j m k')

    # compute third and fourth terms
    g_irk_g_rjm = jnp.einsum('irk, rjm -> ijkm', g_ijk, g_ijk)
    g_irm_g_rjk = jnp.einsum('irm, rjk -> ijkm', g_ijk, g_ijk)

    return g_ijm_k - g_ijk_m + g_irk_g_rjm - g_irm_g_rjk


@manifold_marker.mark(jittable=True)
def fk_riemann_tensor(x: M[jax.Array], metric: MetricFn) -> jax.Array:
    r"""Compute Riemann tensor of the first kind, $R_{ijkl}$

    The Riemann tensor of the first-kind is the index-lowered variant of the Riemann tensor of the second-kind. We
    simply apply an index contraction with the metric tensor to achieve this.

    !!! note
        A more efficient implementation would involve computing this directly by using Christoffel symbols of the
        first kind. The approach implemented here is a little easier to understand. It will not matter once it has
        been compiled down.

    Parameters:
        x: position $p \in M$ at which to evaluate the Riemann curvature tensor
        metric: function defining the metric tensor on the manifold

    Returns:
        Riemann tensor of the first kind at the point $p \in M$
    """

    g_ir = metric(x)
    r_rjkm = sk_riemann_tensor(x, metric)

    return jnp.einsum('ir, rjkm -> ijkm', g_ir, r_rjkm)


@manifold_marker.mark(jittable=True)
def ricci_tensor(x: M[jax.Array], metric: MetricFn) -> jax.Array:
    r"""Compute the Ricci tensor, $R_{ij}$

    The Ricci tensor $R_{ij}$ is a tensor contraction of the Riemann curvature tensor $R^i_{\phantom{i}jkl}$ over the
    first and third indices, precisely

    $$
    R_{ij} = R^k_{\phantom{k}ikj}
    $$

    Parameters:
        x: position $p \in M$ at which to evaluate the Riemann tensor
        metric: function defining the metric tensor on the manifold

    Returns:
        Ricci tensor at the point $p \in M$
    """

    r_kikj = sk_riemann_tensor(x, metric)
    return jnp.einsum('kikj -> ij', r_kikj)


@manifold_marker.mark(jittable=True)
def ricci_scalar(x: M[jax.Array], metric: MetricFn) -> jax.Array:
    r"""Compute the Ricci scalar, $R$.

    The Ricci scalar $R$ yields a single real number which quantifies the curvature on the manifold. It is obtained
    through taking the geometric trace of the Ricci tensor, $R_{ij}$:

    $$
    R = g^{ij} R_{ij}
    $$

    Parameters:
        x: position $p \in M$ at which to evaluate the Riemann tensor
        metric: function defining the metric tensor on the manifold

    Returns:
        Ricci scalar at the point $p \in M$
    """

    contra_g_ij = contravariant_metric_tensor(x, metric)
    r_ij = ricci_tensor(x, metric)

    return jnp.einsum('ij, ij -> ', contra_g_ij, r_ij)


@manifold_marker.mark(jittable=True)
def einstein_tensor(x: M[jax.Array], metric: MetricFn) -> jax.Array:
    r"""Compute the Einstein tensor, $G_{ij}$

    The Einstein tensor, also known as the trace-reversed Ricci tensor is defined as

    $$
    G_{ij} = R_{ij} - \frac{1}{2} g_{ij} R,
    $$

    wjere $R_{ij}$ is the Ricci tensor, and $R$ is the Ricci scalar.

    Parameters:
        x: position $p \in M$ at which to evaluate the Einstein tensor
        metric: function defining the metric tensor on the manifold

    Returns:
        Einstein tensor at the point $p \in M$
    """

    g_ij = metric(x)

    ricci_t = ricci_tensor(x, metric)

    # note: we avoid calls to `contravariant_metric_tensor, ricci_scalar`
    #       to avoid unnecessary computation of the metric and ricci tensor.
    ricci_s = jnp.einsum('ij, ij -> ', jnp.linalg.inv(g_ij), ricci_t)  # type: ignore

    return ricci_t - 0.5 * g_ij * ricci_s


@manifold_marker.mark(jittable=True)
def magnification_factor(x: M[jax.Array], metric: MetricFn) -> jax.Array:
    r"""Compute the magnification factor.

    The magnification factor provides a measure of the local distortion of the distance. It is defined as

    $$
    MF = \sqrt{\lvert g \rvert}
    $$

    Parameters:
        x: position $p \in M$ at which to evaluate the Riemann tensor
        metric: function defining the metric tensor on the manifold

    Returns:
        magnification factor at the point $p \in M$.
    """

    return jnp.sqrt(jnp.linalg.det(metric(x)))
