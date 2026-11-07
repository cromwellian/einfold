<!--
SPDX-FileCopyrightText: 2026 OpenAI

SPDX-License-Identifier: Apache-2.0
-->

# Reuse expensive reductions across axis aliases

The compact executor can share a reduction or contraction value when
its ordered operands, masks and positional domains match an earlier
result in the same invocation. It normalizes the numerical sources of
coordinate views and earlier shared reductions. Every consumer keeps
its own SQL group and validity metadata.

The optional policy considers work over the used dimensions and the
number of factors. It selects at least 2^20 estimated scalar operations
and at least 64 operations per output element. Up to 4,096 identities
are retained. Numerical values use the existing byte-bounded cache,
so eviction can recompute a value from the original expression.
Select the policy with `EINFOLD_DENSE_REDUCTION_VALUES=1`.

## Measurements

The [xarray-sql](https://github.com/xqlsystems/xarray-sql) host ran the
pinned 123.6M Modded-NanoGPT recipe on Apple M5 Ultra, context 1,024,
batch 1, with real FineWeb data and full SQL validation. Coordinate
map sharing was enabled in both arms. [MLX](https://github.com/ml-explore/mlx)
0.32.4 was the numerical GPU backend.

The trace selected 1,081 shared reductions and reduced contraction
builds from 3,797 to 2,717. The optimized graph retained 21,873 nodes
and 1,869 matrix products in both arms. MLX already performs the
corresponding common-expression elimination. Warm training was
592.58 versus 590.08 ms per update in the initial pair and 604.02
versus 602.31 ms with fixed-order lookup sums. Each mean uses updates
11–20. These small differences do not establish a GPU speedup.

The initial checkpoint pair exceeded the 3e-5 absolute and relative
tolerance on one tied embedding weight. With the host's fixed-order
GPU lookup sums, all 147 final parameter and optimizer arrays matched
bit for bit across the sharing and non-sharing runs.

A separate float64 CPU query computes a 512-by-512 Gram matrix twice
under different aliases, joins the results, and returns their sum.
Median time over eight warm calls was 2.14 ms without the pass and
1.33 ms with it. This measures one repeated-contraction query, including
the compact host and scalar Arrow output.

## Verification

A differential test uses repeated 96-by-96 contractions and a distinct
masked contraction. It checks the results against NumPy with both a
one-byte value cache and a larger cache, then changes the source
values. The Python suite passes 286 tests with two optional skips.
The xarray-sql GPU and pinned PyTorch oracle suite passes its focused
cases with the feature enabled.
