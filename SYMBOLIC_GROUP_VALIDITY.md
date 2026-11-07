<!--
SPDX-FileCopyrightText: 2026 OpenAI

SPDX-License-Identifier: Apache-2.0
-->

# Prove SQL group validity without computing counts

Set `EINFOLD_DENSE_SYMBOLIC_MASKS=1` to use dimension metadata when a
group's row/NULL mask depends only on its kept dimensions. Every row in
the reduced Cartesian domain then has the same mask. A nonempty domain
preserves the mask, and an empty domain has no valid contribution.
Neither a dense proof array nor an integer reduction is needed.

The existing full-domain constant-count rule is also selected by this
policy. Count bounds must fit signed int64. Dynamic masks involving
reduced dimensions retain the original count calculation. Numerical
contractions, duplicate contributions, empty-group behavior and NULL
validity remain part of the SQL result. The policy is opt-in.

## Measurements

The [xarray-sql](https://github.com/xqlsystems/xarray-sql) host ran the
pinned historical 123.6M [Modded-NanoGPT](https://github.com/kellerjordan/modded-nanogpt)
recipe on Apple M5 Ultra with [MLX](https://github.com/ml-explore/mlx)
0.32.4, FP32, real FineWeb tokens, batch 1, context 1,024 and full SQL
validation. Coordinate-map sharing and fixed-order lookup sums were
enabled in both training arms. Each run advances twenty updates and
averages updates 11–20.

| Measurement | Control | Symbolic validity |
|---|---:|---:|
| Warm full training update | 602.30 ms | 483.11 ms |
| First update | 7.97 s | 7.13 s |
| Peak process RSS, including checkpoint export | 23.00 GB | 21.95 GB |

The warm speedup is 1.247× in this paired measurement. All 147 final
parameter and optimizer arrays match bit for bit. These measurements
are serial observations, not confidence intervals.

An independent fixed-weight loss/backward probe uses the trained
checkpoint and advancing batches, without optimizer work. Time fell
from 461.43 to 340.66 ms and peak active MLX allocation from 35.04 to
23.74 GB. All 73 gradients pass 3e-5 absolute/relative tolerance.
That probe measures backward computation rather than full training.

## Verification and rejected alternative

Differential SQL tests include empty reduced domains, per-group and
per-row NULL masks, duplicate UNION contributions, COUNT, SUM, AVG,
filters, signed zeros and changed inputs. The Python suite passes 334
tests with two optional skips when the policy is enabled.

A broader prototype replaced remaining count-positive predicates with
Boolean ANY reductions. It measured about 791 ms for loss/backward,
and was removed. Direct symbolic facts are retained; a general mask
reduction rewrite is not part of this change.
