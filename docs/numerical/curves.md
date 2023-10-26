# curves

We provide an implementation of cubic-splines, parameterised by their null-space. The cubic spline is constrained at two end-points, and the basis of the null space is used to parameterise the curve itself.

**Rational for Parameterisation:**

In short, equations for cubic splines form a system of linear homogeneous equations

$$
\mathbf{A} \mathbf{x} = \mathbf{0}
$$

We also know the solution set can be described as

$$
\{ \mathbf{p} + \mathbf{v} : \mathbf{v} \text{ is any solution to } \mathbf{A}\mathbf{x} = \mathbf{0} \}
$$

If we consider elements of the null-basis $\mathbf{\varphi} \in N(A)$, we see

$$
\mathbf{A}\left( \mathbf{x} + \mathbf{\varphi} \right) = \mathbf{0},
$$

so parameterising the cubic spline by the basis of the null-space, we ensure that the equations defining the cubic spline are satisfied.

::: riemax.numerical.curves.CubicSpline
