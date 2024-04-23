import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ...manifold.geometry import inner_product, ricci_scalar, sk_christoffel
from ...manifold.operators import grad as exterior_derivative
from ...manifold.types import M, MetricFn
from .model import EikonalNetwork

# TODO >> @danielkelshaw
#         Consider not vmapping any internals -- return vmapped function.


def construct_eikonal_loss(phi: EikonalNetwork, metric: MetricFn) -> EikonalNetwork:
    def eikonal_loss(params: hk.Params, p: M[jax.Array], q: M[jax.Array]) -> jax.Array:
        fn_phi = jtu.Partial(phi, params, p)
        fn_grad_phi = exterior_derivative(fn_phi, metric)

        grad_phi = fn_grad_phi(q)

        return inner_product(q, grad_phi, grad_phi, metric) - 1.0

    return eikonal_loss


def construct_geodesic_loss(phi: EikonalNetwork, metric: MetricFn) -> EikonalNetwork:
    def geodesic_loss(params: hk.Params, p: M[jax.Array], q: M[jax.Array]) -> jax.Array:
        fn_phi = jtu.Partial(phi, params, p)

        fn_grad_phi = exterior_derivative(fn_phi, metric)
        fn_jacobian_grad_phi = jax.jacfwd(fn_grad_phi)

        grad_phi = fn_grad_phi(q)
        jacobian_grad_phi = fn_jacobian_grad_phi(q)

        # TODO >> @danielkelshaw
        #         Need to check the order of jk here.
        t1 = jnp.einsum('j, kj -> k', grad_phi, jacobian_grad_phi)
        t2 = jnp.einsum('kij, i, j -> k', sk_christoffel(q, metric), grad_phi, grad_phi)

        geodesic_residual = t1 + t2

        return inner_product(q, geodesic_residual, geodesic_residual, metric)

    return geodesic_loss


def curvature_scaling(x: M[jax.Array], alpha: float, metric: MetricFn) -> jax.Array:
    return 1.0 + alpha * jnp.log(1.0 + ricci_scalar(x, metric))


def weight_loss_fn(loss_fn, alpha, metric):
    def weighted_loss(params: hk.Params, p: M[jax.Array], q: M[jax.Array]) -> tuple[jax.Array, dict[str, jax.Array]]:
        residual = loss_fn(params, p, q)
        weighted_residual = residual * curvature_scaling(q, alpha, metric)

        return weighted_residual, dict(residual=residual, weighted_residual=weighted_residual)

    return weighted_loss
