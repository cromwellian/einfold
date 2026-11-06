<!--
SPDX-FileCopyrightText: 2026 OpenAI

SPDX-License-Identifier: Apache-2.0
-->

# Compact coordinate reshapes

The optional compact executor can prove that a static lookup is a
row-major split of one coordinate axis. It then reshapes the value array
and aligns the axes instead of issuing a gather. It can also flatten
an independent one-to-one grouping map without a scatter. The proof
uses index positions, including repetition along an already present axis,
and keeps row presence, SQL NULL flags and counts.

Ten cases compare joins and regrouping with [DataFusion](https://datafusion.apache.org/),
the SQL engine. They cover nonconsecutive labels, repeated and missing keys,
permutations and positions with different maps.

On Apple M5 Ultra, the downstream [xarray-sql](https://github.com/xqlsystems/xarray-sql)
123.6M Modded-NanoGPT training benchmark measured 611.06 to 597.83 ms/update
over eight updates, comparing the last four. A separate twenty-update
comparison measured 606.77 to 599.21 ms over the last ten. All 147
final weights and optimizer buffers passed the existing 3e-5 absolute and
relative tolerance in the layout-only comparison. This uses float32 MLX,
context 1,024, batch 1 and full validation.

The pass is experimental and requires `EINFOLD_DENSE_LAYOUT=1`. A downstream
experiment that also changed indexed gradient accumulation missed the
strict full-model checkpoint tolerance for one weight. The two rewrites
are measured independently.
