<!--
SPDX-FileCopyrightText: 2026 OpenAI

SPDX-License-Identifier: Apache-2.0
-->

# Share coordinate maps across prepared SQL stages

The compact executor can use one coordinate map in a saved forward
statement and its later adjoint. An invocation identifies the key by
its numerical source, positional shape, finite domain proof, and the
ordered destination labels. It reuses offsets and matches even when
SQL gives the dimensions new aliases. A gather can also reuse its
value when the source and positional overlap of its axes agree.
Each consumer keeps its own SQL axes and masks.

If a lookup reads a coordinate that was gathered from that same
ordered domain, the original offsets already specify its group.
This shortcut retains the existing row and value masks. Groups still
receive the original contributions in the original order. The map
cache is local to one invocation and limited by entry count and bytes.
Gather values use the existing computed-value cache and its eviction
policy.

## Measured effect

On an Apple M5 Ultra, the [xarray-sql](https://github.com/xqlsystems/xarray-sql)
host ran the pinned 123.6M Modded-NanoGPT model with real FineWeb data,
batch 1, context 1,024, and full SQL validation. The numerical backend
was [MLX](https://github.com/ml-explore/mlx) 0.32.4. The timing includes
forward, [ddx](https://github.com/xqlsystems/ddx) backward, optimizer
updates, and synchronization. Both paired runs use 20 updates, with
updates 11–20 averaged. The earlier coordinate layout and indexed
UNION experiments were off.

| Pair | Warm control | Warm reuse | First update control | First update reuse |
|---|---:|---:|---:|---:|
| Control first | 607.94 ms | 589.41 ms | 9.86 s | 7.76 s |
| Reuse first | 605.88 ms | 591.42 ms | 9.96 s | 7.88 s |

The compiled trace recorded 547 map hits, 75 elided coordinate
round trips, and 956 shared gather selections. A separate stage
trace observed 622 uses of offsets after their first prepared
statement, including 79 uses in an embedding adjoint of a map
created in a saved forward statement. The full MLX graph fell from
24,466 to 21,873 logical operations. Gather nodes fell from 790 to
201. These counts describe the graph before device dispatch.

The first 147-array checkpoint comparison passed with absolute and
relative tolerances of 3e-5. The reverse-order comparison exceeded
tolerance on one tied embedding weight, with maximum absolute
difference 4.79e-5. Two controls compared with each other passed.
An isolated map and round-trip probe took 615.78 ms. The map and
gather-value probe without the round-trip shortcut took 592.87 ms.
Each isolated probe also exceeded tolerance on one tied embedding
weight. Select the experiment with `EINFOLD_DENSE_INDEX_MAPS=1`.

EinFold's Python suite passed 284 tests with two skips. The xarray-sql
GPU and pinned PyTorch oracle suite passed 106 focused cases. The
differential tests exercise repeated and unmatched keys, differing
coordinate order, NULL masks and a changed batch across calls.
