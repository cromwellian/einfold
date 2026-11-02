<!--
SPDX-FileCopyrightText: 2026 OpenAI
SPDX-License-Identifier: Apache-2.0
-->

# Compact graph optimization

This extends EinFold's optional compact array executor for
[DataFusion](https://datafusion.apache.org/), a SQL engine, and
[ddx](https://github.com/xqlsystems/ddx), an automatic differentiator of SQL.
[xarray-sql](https://github.com/xqlsystems/xarray-sql), a SQL interface to
labeled arrays, supplies a whole training program; Apple's
[MLX](https://github.com/ml-explore/mlx) array framework executes its GPU
arithmetic. Ordinary EinFold SQL extension dispatch is unchanged.

## Changes and guarantees

Previously, independently lowered expressions constructed independent
tensor objects, and reductions hid their dependencies in evaluator
closures. MLX could simplify the resulting device graph, but the host had
already rebuilt and evaluated repeated compact expressions.

The compact executor now interns pure expressions within one invocation.
Common subexpression elimination (CSE) uses the operator, ordered operand
identities, ordered coordinate axes and scalar dtype/bit pattern. It
shares arithmetic, Boolean expressions, rename wrappers and equivalent
contractions. Reductions expose their factor dependencies and include
their complete row/output domains and narrow-integer product-check policy
in their identity. Repeated Boolean predicates and consecutive axis
renames simplify without reordering numerical arithmetic.

Numerical inputs are never interned by value. Current inputs are read for
each invocation, graph metadata expires with it, and the existing
computed-array byte limit still applies. At most 32,768 expressions are
interned; hitting the limit constructs ordinary unshared nodes. Scalar
keys distinguish Boolean/integer/float types and positive/negative zero.
No floating-point reassociation, lower precision, handwritten GPU kernels,
or replacement of ddx's gradient formulas is introduced.

An additional count proof replaces integer COUNT over a full Cartesian
domain with the product of reduced axis lengths. A false mask proves zero.
This retains signed 64-bit bounds, empty-input zero COUNT, NULL SUM/AVG,
and separate row/column validity. Unknown masks remain computations. The
count proof is opt-in because it did not provide a reliable timing gain.

Choose a mode before launching the host process:

| `EINFOLD_DENSE_OPTIMIZE` | Behavior |
|---|---|
| unset or `cse` | Expression sharing, the default |
| `off` or `0` | Reference path without these passes |
| `facts` | Count proofs only |
| `all` | Sharing and count proofs |

Passes require the invocation-local `_ComputedCache` supplied by the
optional host integration. Standalone `DenseProgram.run` without that
execution scope retains its original behavior. These are experimental
compact-executor controls, not a stable general SQL optimizer API.

## Measured effects

Apple M5 Ultra, MLX 0.32.4, float32 with TF32 disabled, November 2, 2026.
The workload is Shakespeare character training: 10,745,088 parameters,
batch 12, context 256, six layers/heads, width 384, dropout .2, exact GELU,
clipped AdamW. All measurements are training, including backward and update.
Each run executes exactly 100 updates with the original 2,000-iteration
learning-rate schedule. Warm means use updates 2–100.

| Training recipe | Reference | CSE | Speedup |
|---|---:|---:|---:|
| Matched CPU dropout masks, validation omitted | 104.30 ms | 97.75 ms | 1.067× |
| Same comparison, repeat | 107.63 ms | 96.59 ms | 1.114× |
| Final source, same repeat reference | 107.63 ms | 97.48 ms | 1.104× |
| GPU dropout, full validation | 94.56 ms | 89.64 ms | 1.055× |

The GPU-dropout pair's first step increased 1.37→2.22 seconds. Including
that first step, its 100 timed updates took 10.73→11.10 seconds, despite
the warm improvement. Longer runs amortize the observed penalty after
about 174 total updates. CPU-mask first steps were 1.39–1.49 seconds for
reference and 1.25–1.36 seconds for CSE. These exclude initial evaluation
and preparation; they are not whole-application startup measurements.

Tracked computed-array cache volume fell 11.11→7.53 GB, about 32%, but
this logical accounting is not physical GPU allocation. Actual MLX peak
allocation varied materially between runs, including 2.70–4.43 GB for the
reference alone; there is no reliable physical-memory reduction claim.
Graph-construction counters include initial evaluation and preparation,
not just a warm training step. GPU contraction counts remained unchanged.

Count proofs alone measured 104.17 ms versus 104.30 ms. Combined passes
measured 94.30 ms with a 3.05-second first step and a higher allocation
peak. An experimental positional contraction matcher across renamed axes
measured 101.47 ms and increased peak allocation; it is not enabled or
included in the implementation. MLX already plans individual contraction
orders, so this change retains that planner.

## Correctness and remaining scope

The Python binding suite passes 118 tests, with two optional skips. The
64 new cases compare DataFusion results across optimization modes, seeds,
empty and nullable inputs, duplicate UNION rows, noncommutative operands,
changing inputs, renamed axes and tiny cache limits. The downstream suite
passes 746 tests, with 277 optional skips and 102 integration deselections.
Five final-source training updates pass the independent nanoGPT/PyTorch
loss, gradient, weight and moment oracle. Matched 100-update checkpoints
agree within 3e-5; observed maximum weight/moment differences are below
2.4e-7.

Native formatting, Clippy and documentation checks pass. The unmodified
Rust `nan_compares_by_its_sign` test still fails on this platform, as it
did on the original baseline. This Python-only change does not fix that
DataFusion behavior difference.

This is the first compiler pass, not completion of a new tensor compiler.
General consumer-aware layout planning, high-level numerical primitive
recognition and joint forward/backward scheduling remain future work.
The full measurements and reproduction harness are in xarray-sql's
`GPU_COMPILER_OPTIMIZATIONS.md` and `benchmarks/apple/compiler-summary.json`.
