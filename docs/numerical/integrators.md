# Integrators

Riemax provides a simple implementation of a number of numerical integrators. However, integration with these relies on standard automatic differentiation. Riemax also provides an implementation of `odeint` which provides custom reverse-mode differentiation in order to compute the adjoint. If you want more options for adjoint-enabled integrators, [Diffrax] is a great place to start. Hopefully, we can add similar functionality here soon...

::: riemax.numerical.integrators

[Diffrax]: https://docs.kidger.site/diffrax/
