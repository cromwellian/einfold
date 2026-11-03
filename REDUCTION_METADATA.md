<!--
SPDX-FileCopyrightText: 2026 OpenAI

SPDX-License-Identifier: Apache-2.0
-->

# Compact reduction and group provenance

[EinFold](https://github.com/xqlsystems/einfold), the tensor optimizer for
SQL, now keeps inspectable reduction metadata in its experimental compact
Python executor. This follows the [graph-sharing work](COMPACT_GRAPH_OPTIMIZATION.md).

SUM/COUNT/AVG contractions retain their original value, factored operands,
row/column mask, joined row axes, kept group axes and repetition count.
MIN/MAX retain their values, masks, axes and expression dependencies.
Grouped UNION ALL retains ordered partial values, masks and coordinate
layouts; Boolean group-presence unions retain their masks and layouts.
Coordinate renames remain explicit views. Inner reduction variables must
be interpreted locally, preserving their scope when a result is renamed.

An optional host `reduction(tensor)` hook can select an equivalent numerical
implementation before invoking the original reduction closure. Returning
`None`, or supplying no hook, keeps that closure. SQL group existence and
column NULL flags remain separate and unchanged. This is private prototype
metadata, not a new stable public API, GPU dependency or differentiation
engine. No array values are cached by content.

This enables structural attention and derivative-block recognition in
[xarray-sql](https://github.com/xqlsystems/xarray-sql), which queries labeled
arrays with SQL. [ddx](https://github.com/xqlsystems/ddx) still generates
the SQL derivatives; hardware kernel selection belongs to the host.
Metadata alone makes no training-speedup claim. The downstream report
records complete training and primitive-selection measurements.

Twenty-four seeded host/no-host cases compare SQL schemas, row counts,
values and NULL flags against [DataFusion](https://datafusion.apache.org/),
the SQL engine. They cover nullable/empty inputs, repeated constants,
renamed MAX values, duplicate UNION contributions and a small computed
cache. The test host chooses some reductions and declines the others.
