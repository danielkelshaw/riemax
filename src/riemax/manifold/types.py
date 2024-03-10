import typing as tp

import jax

type M[T] = T
type TpM[T] = T
type Rn[T] = T

type MetricFn = tp.Callable[[M[jax.Array]], jax.Array]


class TangentSpace[T](tp.NamedTuple):
    r"""Representation of the tangent space on the manifold

    Parameters:
        point: position $p \in M$
        vector: corresponding vector, $v \in T_p M$
    """

    point: M[T]
    vector: TpM[T]
