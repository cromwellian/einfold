<!--
SPDX-FileCopyrightText: 2026 OpenAI

SPDX-License-Identifier: Apache-2.0
-->

# Factor independent row/NULL existence predicates

`EINFOLD_DENSE_FACTOR_MASKS=1` extends the compact executor's symbolic
group-validity analysis. For a conjunction of masks, it joins components
whose reduced axes overlap. Components can share group keys, but their
reduced axes must be disjoint. The existence predicate can then reduce
each component on its smaller domain and combine their Boolean results.
This avoids constructing a count over the combined Cartesian domain for
group presence or aggregate NULL validity.

The rule preserves empty domains, all-NULL groups, independent group keys
and the signed-int64 bound on counts. Numeric SUM/AVG/COUNT expressions and
their reduction order remain unchanged. Masks with coupled reduced axes
retain the original computation. The pass is opt-in and backend independent.

## Measured model behavior

On Apple M5 Ultra with MLX 0.32.4, FP32, the historical 123.6M
[Modded-NanoGPT](https://github.com/kellerjordan/modded-nanogpt) recipe,
batch 1/context 1,024 and full SQL validation, the pass selected 123
factorizations during tracing. It ran alongside symbolic validity,
coordinate-map reuse, fixed-order sums and shared compact embedding
accumulation. Twenty updates averaged updates 11–20 at 412.58 ms, versus
417.24 ms without factorization in the first pair. This small difference
does not establish a reliable warm speedup. All 147 final parameter and
optimizer arrays matched bit for bit against the unoptimized control.

Differential tests compare joined nullable inputs against DataFusion,
including independent mask dimensions, empty domains, filters, products,
SUM/AVG/COUNT and both missing and all-NULL groups. The focused compact
optimizer suite passes 126 tests.
