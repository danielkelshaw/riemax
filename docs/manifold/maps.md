# maps

The `riemax.manifold.maps` module allows the user to define the exponential and log maps on a manifold. These are defined using the geodesic dynamics, defined in `riemax.manifold.geodesic`.

## Types:

```python
type ExponentialMap = tp.Callable[[TangentSpace[jax.Array]], tuple[M[jax.Array], TangentSpace[jax.Array]]]
type LogMap[*Ts] = tp.Callable[[M[jax.Array], M[jax.Array], *Ts], tuple[TangentSpace[jax.Array], bool]]
```


### Exponential Map

Suppose we have a continuous, differentiable manifold, $M$. Given a point $p \in M$, and tangent vector $v \in T_p M$, there exists a unique geodesic $\gamma_v : [0, 1] \rightarrow M$ satisfying $\gamma_v(0) = p$, $\dot{\gamma}_v(0) = v$. The exponential map is defined by $\exp_p(v) = \gamma_v(1)$, or $\exp_p : T_p M \rightarrow M$.

### Log Map

Given two points $p, q \in M$, the $\log$ map provides the tangent-vector which, upon application of the exponential map, transports one point to the other.  The log map is the natural inverse of the exponential map, defined as $\log_p(q) = v$ such that $\exp_p(\log_p(q)) = q$. This mapping is not unique as there may be many tangent-vectors which connect the two points $p, q$.

#### Shooting Solver

Ordinarily, we would consider computing the $\log$ map a two-point boundary value problem. We can approach this using a shooting method, posing the problem: find $v \in T_p M$ such that $\exp_p (v) = q$. We define the residual

$$
r(v) = \exp_p(v) - q,
$$

and use a root-finding technique, such as Newton-Raphson, to obtain a solution. While this remains a principled approach, it is somewhat reliant on having a good initial guess for the solution.


::: riemax.manifold.maps
