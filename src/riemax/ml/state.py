from __future__ import annotations

import pickle
import typing as tp
from pathlib import Path

import haiku as hk
import jax.numpy as jnp
import jax.tree_util as jtu
import optax


class TrainingState(tp.NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

    def save(self, dir: Path) -> None:
        """Save current training state to file.

        Parameters:
            dir: Directory in which to save the state.
        """

        with open(dir / 'arrays.npy', 'wb') as f:
            for x in jtu.tree_leaves(self):
                jnp.save(f, x, allow_pickle=False)

        tree_struct = jtu.tree_map(lambda _: 0, self)
        with open(dir / 'tree.pkl', 'wb') as f:
            pickle.dump(tree_struct, f)

    @classmethod
    def load(cls, dir: Path) -> TrainingState:
        """Load training state from file.

        Parameters:
            dir: Directory from which to load the state.

        Returns:
            Loaded state in the form of a TrainingState.
        """

        with open(dir / 'tree.pkl', 'rb') as f:
            tree_struct = pickle.load(f)

        leaves, treedef = jtu.tree_flatten(tree_struct)
        with open(dir / 'arrays.npy', 'rb') as f:
            flat_state = [jnp.load(f) for _ in leaves]

        return jtu.tree_unflatten(treedef, flat_state)
