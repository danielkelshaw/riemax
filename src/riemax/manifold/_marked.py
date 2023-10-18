from __future__ import annotations

import typing as tp

from .types import MetricFn

type ManifoldFn[*Ts, T] = tp.Callable[[*Ts, MetricFn], T]


class _Marker[*Ts, T](tp.NamedTuple):

    fn: ManifoldFn[*Ts, T]
    jittable: bool


class _ManifoldMarked:

    _self = None

    def __new__(cls) -> _ManifoldMarked:

        if not cls._self:
            cls._self = super().__new__(cls)

        return cls._self

    def __init__(self)-> None:
        self._markers = []

    def __getitem__(self, idx) -> _Marker:
        return self._markers[idx]

    @tp.overload
    def mark(self, fn: None = ..., *, jittable: bool) -> tp.Callable[[ManifoldFn], ManifoldFn]:
        ...

    @tp.overload
    def mark(self, fn: None = ..., *, jittable: tp.Literal[True] = ...) -> tp.Callable[[ManifoldFn], ManifoldFn]:
        ...

    @tp.overload
    def mark(self, fn: None = ..., *, jittable: tp.Literal[False] = ...) -> tp.Callable[[ManifoldFn], ManifoldFn]:
        ...

    @tp.overload
    def mark(self, fn: ManifoldFn) -> ManifoldFn:
        ...

    def mark(self, fn: ManifoldFn | None = None, *, jittable: bool = True) -> ManifoldFn | tp.Callable[[ManifoldFn], ManifoldFn]:

        def _mark(fn: ManifoldFn) -> ManifoldFn:

            _marker = _Marker(fn=fn, jittable=jittable)
            self._markers.append(_marker)

            return fn

        if fn:
            return _mark(fn)

        return _mark


manifold_marker = _ManifoldMarked()
