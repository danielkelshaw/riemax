# Getting Started

???+ example "Note to the reader:"

    This library is very much a work in progress, as is the documentation. Naturally, the contents of this library is subject to change depending on choice of research avenue. At the moment, there exists a solid basis on which to build, but the API may change to improve accessiblity for new users. If there is anything you would like to see in the library, or anything you think that needs to change -- please do let me know.


Riemax is a [JAX] library for Riemannian geometry, providing the ability to define induced metrics and operate on manifolds directly. This includes functionality, such as:

  - Computing geometric quantities on manifolds.
  - Defining operators on manifolds.
  - Tools for geometric statistics.

## Installation

```bash
pip install git+https://github.com/danielkelshaw/riemax
```

Requires Python 3.11+ and JAX 0.4.13+.

## Quick Example

Manifolds can be defined by a function transformation, $\iota: M \rightarrow N$:

```python
import riemax as rx

fn_transformation = rx.fn_transformations.fn_peaks
manifold = rx.Manifold.from_fn_transformation(fn_transformation)
```

and can be used to compute properties on the manifold:

```python
import jax
import jax.numpy as jnp

p = jnp.array([0.0, 0.0])

metric = manifold.metric_tensor(p)
christoffel = manifold.sk_christoffel(p)
ricci_scalar = manifold.ricci_scalar(p)
```
We can also define operators. Given a function $f: M \rightarrow \mathbb{R}$:

```python
from riemax.manifold import M, Rn

def f(p: M[jax.Array]) -> Rn[jax.Array]:
    return jnp.einsum('i -> ', p ** 2)

fn_grad = manifold.grad(f)

# we can define the laplacian explicitly:
fn_laplacian = manifold.div(fn_grad)

# ... or from manifold:
fn_laplacian = manifold.laplace_beltrami(f)
```

We can exploit the ability to compute geometric quantities to compute the exponential map:

```python
dt = 1e-3
n_steps = int(1.0 // dt)

fn_exp_map = rx.manifold.maps.exponential_map_factory(
    integrator=rx.numerical.integrators.odeint,
    dt=dt,
    metric=manifold.metric_tensor,
    n_steps=n_steps
)

p_in_tm = rx.manifold.TangentSpace(p, jnp.array([1.0, 1.0]))
q_in_tm, trajectory = fn_exp_map(p_in_tm)
```

## Next Steps
The examples above show just a fraction of what is possible with Riemax. If this quick start has piqued your interest, please feel free to take a look at the examples to get a better feeling for what is possible with this library.

## Citation

If you found this library to be useful in academic work, then please cite:

```tex
@misc{kelshaw2023riemax
    title = {{Riemax}: differential geometry in {JAX} via automatic differentiation}
    author = {Daniel Kelshaw}
    year = {2023},
    url = {https://github.com/danielkelshaw/riemax}
}
```

## See also: other libraries in the [JAX] ecosystem:

[Optax](https://github.com/google-deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.

[Haiku](https://github.com/google-deepmind/dm-haiku): neural network library.

[Equinox](https://github.com/patrick-kidger/equinox): neural networks.

[Diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers.


[JAX]: https://github.com/google/jax
