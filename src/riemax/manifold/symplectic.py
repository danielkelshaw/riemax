from __future__ import annotations

import functools as ft
import typing as tp

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ..numerical.integrators import _merge_states
from .geometry import contravariant_metric_tensor
from .types import MetricFn, TangentSpace


class SymplecticGeodesicState(tp.NamedTuple):
    """PyTree for Symplectic Geodesic State.

    Parameters:
        q: position on the geodesic
        p: conjugate momenta on the co-tangent space
    """

    q: jax.Array
    p: jax.Array

    @classmethod
    def from_lagrangian(cls, state: TangentSpace[jax.Array], metric: MetricFn) -> SymplecticGeodesicState:
        """Build Hamiltonian symplectic state from given Lagrangian state.

        Parameters:
            state: Geodesic state in Lagrangian coordinates.
            metric: Function used to evaluate the metric.

        Returns:
            Symplectic state in Hamiltonian coordinates.
        """

        conjugate_momenta = jnp.einsum('...ij, ...j -> ...i', metric(state.point), state.vector)
        return cls(q=state.point, p=conjugate_momenta)

    def to_lagrangian(self, metric: MetricFn) -> TangentSpace[jax.Array]:
        """Convert intrinstic Hamiltonian coordinates to Lagrangian coordinates.

        Parameters:
            metric: Function used to evaluate the metric.

        Returns:
            Geodesic state in Lagrangian coordinates.
        """

        velocity = jnp.einsum('...ij, ...j -> ...i', contravariant_metric_tensor(self.q, metric), self.p)
        return TangentSpace(point=self.q, vector=velocity)


class PhaseDoubledSymplecticGeodesicState(tp.NamedTuple):
    """PyTree for the phase-doubled Symplectic Geodesic State

    Parameters:
        q: position on the geodesic
        p: conjugate momenta on the co-tangent space
        x: phase-doubled position on the geodesic
        y: phase-doubled conjugate momenta on the co-tangent space
    """

    q: jax.Array
    p: jax.Array
    x: jax.Array
    y: jax.Array

    @classmethod
    def from_symplectic(cls, state: SymplecticGeodesicState) -> PhaseDoubledSymplecticGeodesicState:
        """Build phase-doubled symplectic state from given symplectic state.

        Parameters:
            state: Symplectic state in a single phase-space.

        Returns:
            Phase-doubled symplectic state, replicating state in new phase-space.
        """

        return cls(q=state.q, p=state.p, x=state.q, y=state.p)

    def to_symplectic(self) -> SymplecticGeodesicState:
        """Transform phase-doubled symplectic state to a single-phase symplectic state.

        Returns:
            Single-phase symplectic state -- removing phase-doubling.
        """

        return SymplecticGeodesicState(q=self.q, p=self.p)


def hamiltonian(q: jax.Array, conjugate_momenta: jax.Array, metric: MetricFn) -> jax.Array:
    """Computes the Hamiltonian of the state.

    Parameters:
        q: position on the geodesic
        conjugate_momenta: conjugate momenta of the geodesic path
        metric: function defining the metric tensor on the manifold

    Returns:
        hamiltonian of the geodesic
    """

    fn_contra_gx = jtu.Partial(contravariant_metric_tensor, metric=metric)
    return 0.5 * jnp.einsum('ij, i, j -> ', fn_contra_gx(q), conjugate_momenta, conjugate_momenta)


def _phi_ha(
    state: PhaseDoubledSymplecticGeodesicState, dt: float, metric: MetricFn
) -> PhaseDoubledSymplecticGeodesicState:
    """Compute first phase-map.

    Parameters:
        state: current state of the symplectic geodesic
        dt: time-step used for the integration
        metric: function defining the metric tensor on the manifold

    Returns:
        updated geodesic state
    """

    fn_hamiltonian = jtu.Partial(hamiltonian, metric=metric)

    # we compute updates for H(q, y)
    dq_h, dy_h = jax.jacfwd(fn_hamiltonian, argnums=(0, 1))(state.q, state.y)

    p_updated = state.p - dt * dq_h
    x_updated = state.x + dt * dy_h

    return PhaseDoubledSymplecticGeodesicState(state.q, p_updated, x_updated, state.y)


def _phi_hb(
    state: PhaseDoubledSymplecticGeodesicState, dt: float, metric: MetricFn
) -> PhaseDoubledSymplecticGeodesicState:
    """Compute second phase-map.

    Parameters:
        state: current state of the symplectic geodesic
        dt: time-step used for the integration
        metric: function defining the metric tensor on the manifold

    Returns:
        updated geodesic state
    """

    fn_hamiltonian = jtu.Partial(hamiltonian, metric=metric)

    dx_h, dp_h = jax.jacfwd(fn_hamiltonian, argnums=(0, 1))(state.x, state.p)

    q_updated = state.q + dt * dp_h
    y_updated = state.y - dt * dx_h

    return PhaseDoubledSymplecticGeodesicState(q_updated, state.p, state.x, y_updated)


def _phi_hc(state: PhaseDoubledSymplecticGeodesicState, dt: float, omega: float) -> PhaseDoubledSymplecticGeodesicState:
    """Compute third phase-map.

    Parameters:
        state: current state of the symplectic geodesic
        dt: time-step used for the integration
        omega: strength of the constraint between the phase-split copies

    Returns:
        updated geodesic state
    """

    r_mat = jnp.array(
        [
            [jnp.cos(2.0 * omega * dt), jnp.sin(2.0 * omega * dt)],
            [-jnp.sin(2.0 * omega * dt), jnp.cos(2.0 * omega * dt)],
        ]
    )

    sum_vec = jnp.array([state.q + state.x, state.p + state.y])
    dif_vec = jnp.array([state.q - state.x, state.p - state.y])

    q, p = 0.5 * (sum_vec + jnp.einsum('ij, jk -> ik', r_mat, dif_vec))
    x, y = 0.5 * (sum_vec - jnp.einsum('ij, jk -> ik', r_mat, dif_vec))

    return PhaseDoubledSymplecticGeodesicState(q, p, x, y)


def _yoshida_triple_jump_constants(n: int) -> tuple[float, float]:
    """Compute z0, z1 constants for Yoshida triple-jump.

    Parameters:
        n: constant defining order of integration

    Returns:
        z0: first phase-map constant
        z1: second phase-map constant
    """

    const = 2.0 ** (1.0 / (2.0 * n + 1.0))

    z1 = 1.0 / (2.0 - const)
    z0 = -const * z1

    return z0, z1


SymplecticUpdator: tp.TypeAlias = tp.Callable[
    [PhaseDoubledSymplecticGeodesicState, float, float, MetricFn], PhaseDoubledSymplecticGeodesicState
]


def second_order_dynamics(
    state: PhaseDoubledSymplecticGeodesicState, dt: float, omega: float, metric: MetricFn
) -> PhaseDoubledSymplecticGeodesicState:
    """Conduct time-step using second-order dynamics.

    Parameters:
        state: current state of the symplectic integrator
        dt: time-step used for the integration
        omega: strength of the constraint between the phase-split copies
        metric: function defining the metric tensor on the manifold

    Returns:
        state: time-stepped, phase-doubled symplectic geodesic state
    """

    fn_phi_ha = jtu.Partial(_phi_ha, dt=dt, metric=metric)
    fn_phi_hb = jtu.Partial(_phi_hb, dt=dt, metric=metric)
    fn_phi_hc = jtu.Partial(_phi_hc, dt=dt, omega=omega)

    state = fn_phi_ha(state=state)
    state = fn_phi_hb(state=state)
    state = fn_phi_hc(state=state)
    state = fn_phi_hb(state=state)
    state = fn_phi_ha(state=state)

    return state


def construct_nth_order_dynamics(n: int):
    """Construct nth order symplectic dynamics.

    !!! note "Recursive definition:"
        Function works recursively, producing additional phase-maps as required.

    Parameters:
        n: order of integration to produce

    Returns:
        function to compute nth order dynamics
    """

    if not n % 2 == 0:
        raise ValueError('Only works for even n.')

    @ft.lru_cache()
    def _construct(n: int):
        if n == 2:
            return second_order_dynamics

        def nth_order_dynamics(state, dt, omega, metric):
            _n = (n - 2) // 2
            z0, z1 = _yoshida_triple_jump_constants(n=_n)

            nmt_dynamics = _construct(n - 2)
            fn_phi_a = jtu.Partial(nmt_dynamics, dt=(z1 * dt), omega=omega, metric=metric)
            fn_phi_b = jtu.Partial(nmt_dynamics, dt=(z0 * dt), omega=omega, metric=metric)

            state = fn_phi_a(state=state)
            state = fn_phi_b(state=state)
            state = fn_phi_a(state=state)

            return state

        return nth_order_dynamics

    return _construct(n)


class SymplecticParams(tp.NamedTuple):
    """Contained for parameters of symplectic integration

    Parameters:
        metric: function defining the metric tensor on the manifold
        dt: time-step used for the integration
        omega: strength of the constraint between the phase-split copies
        n_steps: number of steps to integrate for
    """

    metric: MetricFn
    dt: float = 1e-3
    omega: float = 1e-2
    n_steps: int = int(1e3)


PhaseDoubledSymplecticIntegrator: tp.TypeAlias = tp.Callable[
    [SymplecticParams, PhaseDoubledSymplecticGeodesicState],
    tuple[PhaseDoubledSymplecticGeodesicState, PhaseDoubledSymplecticGeodesicState],
]
LagrangianSymplecticIntegrator: tp.TypeAlias = tp.Callable[
    [SymplecticParams, TangentSpace[jax.Array]], tuple[TangentSpace[jax.Array], TangentSpace[jax.Array]]
]


def construct_nth_order_symplectic_integrator(n: int) -> PhaseDoubledSymplecticIntegrator:
    """Construct symplectic integrator of the nth order.

    Parameters:
        n: order of integration required

    Returns:
        integrator for phase-doubled symplectic state
    """

    def _nth_order_symplectic_integrator(
        symplectic_params: SymplecticParams, initial_state: PhaseDoubledSymplecticGeodesicState
    ) -> tuple[PhaseDoubledSymplecticGeodesicState, PhaseDoubledSymplecticGeodesicState]:
        """Integrator using nth order symplectic dynamics.

        Parameters:
            symplectic_params: parameters for the symplectic integration
            initial_state: state at t=0
        """

        nth_order_dynamics = construct_nth_order_dynamics(n)
        fn_updator = jtu.Partial(
            nth_order_dynamics, dt=symplectic_params.dt, omega=symplectic_params.omega, metric=symplectic_params.metric
        )

        def _single_step(
            state: PhaseDoubledSymplecticGeodesicState, _: None
        ) -> tuple[PhaseDoubledSymplecticGeodesicState, PhaseDoubledSymplecticGeodesicState]:
            return fn_updator(state=state), state

        final_state, preceding_states = jax.lax.scan(_single_step, initial_state, None, symplectic_params.n_steps)
        full_state = _merge_states(preceding=preceding_states, final=final_state)

        return final_state, full_state

    return _nth_order_symplectic_integrator


def _phase_doubled_to_lagrangian_integrator(
    symplectic_integrator: PhaseDoubledSymplecticIntegrator,
) -> LagrangianSymplecticIntegrator:
    """Provide an interface to allow passing Lagrangian coordinates.

    Parameters:
        symplectic_integrator: symplectic integrator, operating with phase-doulbled state

    Returns:
        lagrangian_interface: a lagrangian interface to the integrator
    """

    def lagrangian_interface(
        symplectic_params: SymplecticParams, initial_state: TangentSpace[jax.Array]
    ) -> tuple[TangentSpace[jax.Array], TangentSpace[jax.Array]]:
        """Lagrangian integrator using nth order symplectic dynamics.

        Parameters:
            symplectic_params: parameters for the symplectic integration
            initial_state: state at t=0
        """

        # convert lagrangian to phase-doubled symplectic
        symplectic_initial_state = SymplecticGeodesicState.from_lagrangian(
            initial_state, metric=symplectic_params.metric
        )
        pd_symplectic_initial_state = PhaseDoubledSymplecticGeodesicState.from_symplectic(symplectic_initial_state)

        # conduct integration with phase-doubled symplectic state
        pd_symplectic_final_state, pd_symplectic_full_state = symplectic_integrator(
            symplectic_params, pd_symplectic_initial_state
        )

        # convert phase-doubled symplectic state back to Lagrangian
        symplectic_final_state, symplectic_full_state = map(
            PhaseDoubledSymplecticGeodesicState.to_symplectic, (pd_symplectic_final_state, pd_symplectic_full_state)
        )
        final_state, full_state = map(
            jtu.Partial(SymplecticGeodesicState.to_lagrangian, metric=symplectic_params.metric),
            (symplectic_final_state, symplectic_full_state),
        )

        return final_state, full_state

    return lagrangian_interface
