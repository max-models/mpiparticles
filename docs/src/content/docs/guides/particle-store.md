---
title: Particle store
description: Component-major marker storage with growth, compaction, sorting, checkpoints and a MarkerSet view.
sidebar:
  order: 1
---

`ParticleStore(ndim, nvel, *, capacity=16, dtype=float64)` holds the markers of one rank. All
arrays are made through `cunumpy`, so they are NumPy arrays on the CPU and CuPy arrays on a GPU.

| Accessor     | Shape                 | Content                                   |
| ------------ | --------------------- | ----------------------------------------- |
| `positions`  | `(ndim, capacity)`    | component-major positions                 |
| `velocities` | `(nvel, capacity)`    | component-major velocities (`nvel >= ndim`) |
| `weights`    | `(capacity,)`         | weights, 1 for unused columns             |
| `alive`      | `(capacity,)` bool    | whether a column holds a live marker      |
| `ids`        | `(capacity,)` int64   | stable marker ids                         |

The accessors return the full-capacity arrays; the first `length` columns (the high-water mark)
are in use, `num_alive` of them alive. Kernels take strided in-place views of
`positions[:, :length]` and friends.

## Growth

`add` appends alive markers (positions `(ndim, n)`, velocities `(nvel, n)`, optional weights
and ids) and returns their ids. When the capacity is exceeded the storage grows by doubling,
which allocates new arrays: **views taken earlier go stale**. Call `reserve(n)` once before
taking views for a push, and take the views again after any `add`, `reserve`, `restore` or `migrate`.
Storage never reallocates during a push.

## Dead markers and compaction

`deactivate(mask)` marks columns dead (`mask` covers `[:length]`); they keep their data and stay
in storage, so loops over `[:length]` keep working. `compact()` moves the alive markers to the front
in their original order. `compact_if_needed(max_dead_fraction=0.25)` compacts once the dead fraction
exceeds the limit and returns whether it did. The store never compacts by itself, so the caller
decides when views may move. `remove(index)` removes one column and `clear()` empties the store.

## Sorting

`sort_by_key(keys)` takes one integer key per stored column (for example the cell index), reorders
all fields by it with a stable sort and drops the dead markers. Ids move as int64 and are never
viewed as floats.

## Alive markers

`active_positions()`, `active_velocities()`, `active_weights()` and `active_ids()` return copies of
the alive markers; `set_active_positions`, `set_active_velocities` and `set_active_weights` write
them back.

## Checkpoints

`snapshot()` returns a `StoreSnapshot`: a copy of every column in use, alive or dead, and the id
counter. `restore(snapshot)` puts it back, growing the storage if needed.

```python
snapshot = store.snapshot()
# ... a trial step that may be rejected ...
store.restore(snapshot)
```

## The MarkerSet view

`store.as_marker_set()` returns a marker-major view that satisfies
[plasmacoll](https://github.com/max-models/plasmacoll)'s `MarkerSet` protocol (mpiparticles does not import
plasmacoll): `positions` is `(N, ndim)`, `velocities` `(N, nvel)`, `weights` `(N,)`, all transposed views
of the storage, not copies. Updates such as the velocity changes of a collision land in the store.
`remove(mask)` and `add(positions, velocities, weights)` work on the store. The view needs a
store without dead rows, so it compacts when it is made and again when it is read after markers were
deactivated.
