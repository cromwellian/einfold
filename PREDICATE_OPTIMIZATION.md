<!--
SPDX-FileCopyrightText: 2026 OpenAI
SPDX-License-Identifier: Apache-2.0
-->

# Boolean normalization and count-presence proofs

The optional compact executor lowers [DataFusion](https://datafusion.apache.org/)
SQL plans to arrays. [ddx](https://github.com/xqlsystems/ddx), an automatic
differentiator of SQL, produces repeated `COUNT(...) > 0` expressions and
Boolean CASE expressions to preserve row and NULL semantics in gradients.
Previously these expressions could remain nested even when simpler
equivalent predicates were available.

## Transformations

Boolean normalization folds literal negation, rewrites double negation
through a Boolean cast, removes duplicate Boolean casts and simplifies
CASE expressions with opposite TRUE/FALSE branches. The cast preserves
the result type; numeric CASE branches are not converted to Boolean.

An optional count-presence fact applies when an integer COUNT's row mask
depends only on kept dimensions. It is constant along every reduced
dimension, so COUNT is the mask times their extent product. If that product
is positive and fits signed 64-bit COUNT, `COUNT > 0` equals the mask.
An empty reduced domain gives FALSE in this same eligible case. Numerical
COUNT values are unchanged. Axis renames and gathers transform the fact
along with the count.

Unknown masks that vary along reduced dimensions and possible count
overflows retain their original predicate. Dynamic NULL masks, filters,
empty aggregate behavior and bag multiplicity remain. No numerical
finiteness assumption, floating reassociation, dtype reduction or
replacement of the ddx gradient is introduced.

Set `EINFOLD_DENSE_PREDICATES` before launching the host:

| Value | Behavior |
|---|---|
| unset or `boolean` | Boolean normalization, the selected default |
| `off` or `0` | Disable these new predicate rewrites |
| `presence` | Count-presence facts only |
| `all` | Both passes |

`EINFOLD_DENSE_OPTIMIZE=off` also disables predicate rewrites unless their
mode is explicitly supplied. Invocation-local scope and existing graph/cache
limits are unchanged. As with the earlier compact passes, these controls
are private prototype controls, not a new unconditional SQL optimizer API.

## Training measurements and defaults

Apple M5 Ultra, public MLX 0.32.4 float32, TF32 disabled, November 2, 2026.
The host is [xarray-sql](https://github.com/xqlsystems/xarray-sql), a SQL
interface to labeled arrays, training the 10,745,088-parameter Shakespeare
character model: batch 12, context 256, six layers/heads, width 384, dropout
.2, exact GELU, clipped AdamW. GPU dropout and full validation remain enabled.
Each run progresses through exactly 100 updates with the original schedule.
Warm means use updates 2–100; initial evaluation and input/I/O costs are
outside that timer.

| Predicate mode, before host layout optimization | Warm update |
|---|---:|
| Off | 87.56 ms |
| Boolean normalization | 82.69 ms |
| Count-presence proofs | 83.63 ms |
| Both | 79.11 ms |

The full-run counters record 550 Boolean normalizations and 12,227 optional
presence proofs. They include evaluation and graph construction, not just
one warm step. GPU contraction dispatch counts were unchanged; these
observations do not identify individual kernel timing contributions.

The host separately replaces identity-layout gradient scatter-adds with
direct additions. With that change, the selected Boolean-only default
measured 63.91 ms initially and 61.27 ms in a repeated pair, versus
87.56/85.58 ms references: **1.37–1.40× faster warm training**. The
repeated first step was 1.06 versus 1.14 seconds, but cold timings varied
considerably across experiments and are not a guaranteed startup gain.

Count-presence proofs with direct accumulation measured 60.05–61.88 ms,
but raised observed MLX peak allocation to 7.16 GB versus 4.36 GB for the
selected default. The small additional warm benefit did not justify that
allocation cost as a default. Whole-step pointwise inlining was also
tested downstream and discarded after a controlled replay regression.

## Verification

The Python binding suite passes **214 tests**, two optional skips. The 96
new seeded cases compare DataFusion SQL across all modes, kept/reduced/mixed
NULL masks, empty domains, filtering, typed Boolean CASE results, nested
counts and aliases. Downstream adds 16 CPU/GPU cases with changing numeric
NULL predicates and empty groups, and its full suite passes **770 tests**.
Five default-optimized updates pass the original nanoGPT/PyTorch oracle
for loss, raw gradients, weights and moments. Matched 100-update checkpoints
agree within 3e-5, with maximum observed differences below 2.4e-7.

Native formatting, Clippy and documentation checks pass. The pre-existing
Rust `nan_compares_by_its_sign` test still fails on this platform; excluding
it leaves 72 passing tests and four ignored. No Rust code changed.
The downstream `GPU_PREDICATE_OPTIMIZATIONS.md` report and its per-update
evidence distinguish these passes from the host's independent layout change.
