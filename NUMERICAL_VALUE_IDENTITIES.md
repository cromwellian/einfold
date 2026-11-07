<!--
SPDX-FileCopyrightText: 2026 OpenAI

SPDX-License-Identifier: Apache-2.0
-->

# Share positional values across the complete prepared invocation

Enable `EINFOLD_DENSE_VALUE_IDENTITIES=1` to share numerical producers
across axis aliases in forward and backward statements. Each consumer
retains its own coordinate labels, row/NULL masks and reduction metadata.
The pass neither reorders arithmetic nor replaces differentiation.

Pure pointwise keys retain operation, operand order, extents and positional
axis roles. Sum/contraction keys retain every operand, mask, complete row
and group domains, repeated extents and integer-product provenance. MIN/MAX
also retain operation and masks. An indexed sum can share only a retained
lookup offset proven to use the same ordered domain, with matching input,
row layout and count bounds. Ordered UNION keys retain each branch's value
and mask. Unproved or differently ordered domains keep their own producers.

The lookup rules use `EINFOLD_DENSE_INDEX_MAPS=1`. All identities expire at
the end of the invocation; new input arrays are not matched by their values.
Identity maps are bounded separately from numerical storage. Shared values
remain in the existing byte-bounded LRU and recompute after eviction.

## Measurements

The [xarray-sql](https://github.com/xqlsystems/xarray-sql) host ran the
pinned historical 123.6M [Modded-NanoGPT](https://github.com/kellerjordan/modded-nanogpt)
recipe on Apple M5 Ultra, [MLX](https://github.com/ml-explore/mlx) 0.32.4,
FP32, real FineWeb, batch 1/context 1,024 and full validation. Both arms
enable symbolic validity, coordinate reuse, shared compact accumulation
and fixed-order sums. Timings average updates 11–20 of twenty advancing
updates from scratch.

Pointwise and contraction identities alone selected thousands of shared
values but retained about 418 ms/update. Extending the same mechanism to
extrema and indexed sums reduced the first complete run to 232.93 ms;
an independent repeat measured 233.98 ms. Disabling indexed identities
returned time to 417.95 ms. Disabling extrema identities retained 234.95 ms.
The indexed family accounts for the large measured gain. These are paired
observations, not confidence intervals.

All 147 final parameter and optimizer arrays match the control bit for bit.
The trace reduced contraction construction from 3,644 to 2,137, and scatter
construction from 418 to 197. These are construction counters, not GPU
kernel counts. The separate mask and accumulation optimizations are not
attributed to this pass.

## Verification

Differential tests cover repeated and changed sources, NULLs, joins,
duplicate contributions, labels that differ between aliases and snapshots.
Two prepared queries reuse a physical value under different coordinates
and validity masks with one-byte and larger caches. The Python suite passes
348 tests with two optional skips. Full model comparisons include every
parameter and optimizer buffer; no intermediate arrays are exported in the
timed training steps.

The host suite passes 884 tests with 277 optional skips and 102 integration
deselections. Rust formatting, Clippy, documentation and REUSE checks pass.
The full Rust run reproduces the existing `nan_compares_by_its_sign` failure;
remaining workspace tests pass when it is excluded. No Rust implementation
or differentiation rule changes are included.
