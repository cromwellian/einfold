<!--
SPDX-FileCopyrightText: 2026 OpenAI

SPDX-License-Identifier: Apache-2.0
-->

# Indexed partial provenance

The compact executor retains the input values, key and local axes for a
grouped scatter. The values already include the original row and NULL
masks. This provides enough information for a downstream numerical host
to group repeated keys in a compact buffer and add the resulting partial
to an existing UNION accumulator. The metadata leaves aggregate values,
keys and validity flags available through their existing paths.

The [xarray-sql](https://github.com/xqlsystems/xarray-sql) Apple GPU
experiment applied this to several indexed contributions to the tied
embedding gradient in 123.6M-parameter Modded-NanoGPT. On M5 Ultra, the
isolated indexed variant measured 527.5 ms/update versus 606.8–617.9 ms
for the generic SQL reference over 20 updates. All 147 checkpoint arrays
passed the existing 3e-5 tolerance. The measurements apply to the
downstream implementation and its float32 policy. The metadata alone
does not select a kernel.
