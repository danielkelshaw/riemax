# symplectic

The geodesic equation is usually solved in the classic Lagrangian form, however it also admits a Hamiltonian. This hamiltonian is non-separable, meaning that standard methods for integration are not feasible. Recently, an approach for integration of non-separable Hamiltonians has been developed, doubling the phase-space and evolving states in parallel.[^1][^2][^3] Riemax provides an implementation of this approach for defining dynamics of geodesics upto arbitrary orders of integration.

!!! warning "Additional documentation required."

    This documentation requires further explanation of the method for integration of non-separable Hamiltonians. This page will be updated in future iterations of the documentation to make the process as clear as possible.

[^1]:
    Christian, Pierre, and Chi-kwan Chan. ‘FANTASY: User-Friendly Symplectic Geodesic Integrator for Arbitrary Metrics with Automatic Differentiation’. The Astrophysical Journal 909, 2021. <a href="https://doi.org/10.3847/1538-4357/abdc28">https://doi.org/10.3847/1538-4357/abdc28</a>

[^2]:
    Tao, Molei. ‘Explicit Symplectic Approximation of Nonseparable Hamiltonians: Algorithm and Long Time Performance’. Physical Review E 94, 2016. <a href="https://doi.org/10.1103/PhysRevE.94.043303">https://doi.org/10.1103/PhysRevE.94.043303</a>

[^3]:
    Yoshida, Haruo. ‘Construction of Higher Order Symplectic Integrators’. Physics Letters A, 1990. <a href="https://doi.org/10.1016/0375-9601(90)90092-3">https://doi.org/10.1016/0375-9601(90)90092-3</a>


::: riemax.manifold.symplectic
