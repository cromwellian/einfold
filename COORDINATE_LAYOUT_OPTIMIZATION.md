<!--
SPDX-FileCopyrightText: 2026 OpenAI

SPDX-License-Identifier: Apache-2.0
-->

# Compact lowering for cyclic coordinate maps

The compact executor can split a flat axis when a coordinate join proves
that the lookup visits each source position once. It now also recognizes a
cyclic shift on each split axis. The first mapped offset specifies the
shifts. The proof checks every position against the resulting map. The
forward uses a reshape and one small lookup per shifted axis. A grouping
on the same map applies the inverse shifts and flattens the result.

A key can mention an independent grouping axis when its offsets are
constant along that axis. Each independent group then has a single
contribution for every position in the key domain. The original SQL
row-presence and value-validity masks still determine whether a row
contributes. Arbitrary permutations, varying or repeated offsets, and
device-computed keys use the general gather and grouped scatter.

Select `EINFOLD_DENSE_LAYOUT=extended` to enable the experiment.
`EINFOLD_DENSE_LAYOUT=1` selects the earlier ordered reshape proof.
The proofs use ordered coordinate positions and run on arbitrary queries
with compatible maps.

## Measured on Apple M5 Ultra

The [xarray-sql](https://github.com/xqlsystems/xarray-sql) host compiles a
full training step with [ddx](https://github.com/xqlsystems/ddx) for SQL
differentiation and [MLX](https://github.com/ml-explore/mlx) for GPU
execution. The experiment used the pinned historical 123.6M
[Modded-NanoGPT](https://github.com/kellerjordan/modded-nanogpt) recipe,
real FineWeb data, batch 1, context 1,024, and full SQL validation.
The candidate ran as a Python source overlay on the same native wheel
as the control. Both runs enabled the layout pass.

| Warm timing | Existing proof | Extended proof |
|---|---:|---:|
| 20 updates, last 10 averaged | 601.16 ms | 594.00 ms |
| Reverse order, 12 updates, last 6 averaged | 608.29 ms | 597.75 ms |
| Final opt-in mode, 20 updates | 601.54 ms | 598.76 ms |

The first update in the twenty-update pair took 9.71 seconds with the
existing proof and 8.85 seconds with the extension. Peak process RSS
through checkpoint export was 25.72 GB and 22.81 GB, respectively.
The compiled graph had 22,801 logical operations with the existing
proof and 21,696 with the extension. Scatter-sum nodes fell from 401 to
225. The graph's matmul count remained 1,860.

At 20 updates, the initial extended run passed against the repeated
control and missed the original control on one tied embedding weight.
The final opt-in mode missed each control on one tied embedding weight,
with maximum absolute differences of 5.91e-5 and 7.82e-5. Comparisons
cover all 147 checkpoint arrays with absolute and relative tolerances
of 3e-5. The separate cyclic-shift-only probe passed against the first
control and ran at 599.13 ms. The separate shared-axis-only probe
ran at 599.85 ms and missed two embedding weights against that control.

The SQL differential tests exercise a reshape and its inverse group,
with nonconsecutive labels, NULL values, shared axes, repeated keys,
and cyclic shifts in two dimensions. EinFold's Python suite passed
282 tests with two skips. The xarray-sql GPU and upstream-oracle
suite passed 105 focused tests with the extension enabled.
Rust formatting, Clippy, documentation, and license checks passed.
The Rust suite reproduced the existing `nan_compares_by_its_sign`
failure. The other Rust tests passed when that case was excluded.
