import typing as tp

import haiku as hk
import jax
import optax

from .state import TrainingState


def construct_update[**P, T, S](
    loss_fn: tp.Callable[tp.Concatenate[hk.Params, P], tuple[T, S]], optimiser: optax.GradientTransformation
) -> tp.Callable[tp.Concatenate[TrainingState, P], tuple[tuple[T, S], TrainingState]]:
    """Construct an update function from a given loss function.

    Parameters:
        loss_fn: loss function used for optimisation, must return aux.
        optimiser: optimiser used for the training process.

    Returns:
        update_fn: function used to update the state of the training.
    """

    def update_fn(state: TrainingState, *args: P.args, **kwargs: P.kwargs) -> tuple[tuple[T, S], TrainingState]:
        aux_loss, grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=0)(state.params, *args, **kwargs)

        updates, opt_state = optimiser.update(grads, state.opt_state, params=state.params)
        params = tp.cast(hk.Params, optax.apply_updates(state.params, updates))

        return aux_loss, TrainingState(params, opt_state)

    return update_fn


def construct_initialiser[**P, T, S](
    transform_init: tp.Callable[tp.Concatenate[jax.Array, P], hk.Params],
    optimiser: optax.GradientTransformation,
) -> tp.Callable[tp.Concatenate[jax.Array, P], TrainingState]:
    """Construct initialised for a TrainingState.

    Parameters:
        transform_init: initialiser provided by the Haiku transform.
        optimiser: optimiser used for the training process.

    Returns:
        _initialise: a function which returns an initialised TrainingState.
    """

    def _initialise(key: jax.Array, *args: P.args, **kwargs: P.kwargs) -> TrainingState:
        params = transform_init(key, *args, **kwargs)
        opt_state = optimiser.init(params)

        return TrainingState(params, opt_state)

    return _initialise
