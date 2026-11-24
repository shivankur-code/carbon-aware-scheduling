# Assumptions register

Every model parameter must have a **source** or be tagged **ASSUMPTION**, with the range
used in sensitivity analysis. The pilot values are all placeholders.

| Parameter | Pilot value | Status | Sweep range | Source to find / notes |
|---|---|---|---|---|
| Embodied carbon per GPU incl. server share | 1000 kgCO2e | ASSUMPTION (upper-middle) | 220–2500 | EcoServe Sec 6.5 uses 120 kg per GPU + 800 kg per host (≈220 kg/GPU at 8 GPUs/host). **Checked: conclusions hold at 220 kg** — clock locking saves 18/24/25% (low/mid/high grid) vs 8/20/23% at 1000 kg; 95%-chance plan premium 1.4% vs 1.9%. |
| Average active power per GPU (incl. server share) | 0.40 kW | ASSUMPTION | 0.25–0.9 | Vendor TDP; MLPerf Inference power results |
| Spare idle power | 0.10 kW | ASSUMPTION | 0–0.2 | Spares may be powered off (cold), which trades availability against energy |
| PUE | 1.2 | ASSUMPTION | 1.1–1.5 | Public hyperscaler sustainability reports |
| Perf/W doubling period | 3.5 y | Matches EcoServe (Sec 6.5, citing Sun et al. 2019) [V] | 2–5 y | Verify in EcoServe PDF; MLPerf generations |
| Weibull shape k | 2.0 | **Weakened by evidence.** Cui et al. (SC'25) find the GPU hazard rate "exhibits no clear temporal trends during the operational period" (A100 over 2.5 y, calendar time, not age-stratified). So k ≈ 1 is the data-backed default for years 0–2.5; wear-out beyond that is unobserved. | 1 – 3; k > 1 only for ages > 2.5 y | Cui et al. 2025 [V] |
| Weibull scale | 5/8/12 y | ASSUMPTION for *hard* (replacement) failures. Field replacements are rare: 4 A100s over ~1,098 GPU-years (0.36%/y) and 2 H100s over ~243 GPU-years (0.82%/y), derived from Cui et al. | 0.2–10 %/y | Cui et al. 2025 [V, derived] |
| **Critical GPU error rate** (interrupts the node) | — | **Sourced:** 13.5 errors per A100 GPU-year (14,821 / 1,098) and 7.5 per H100 GPU-year (1,821 / 243), derived from Cui et al. Table 1 / Sec 4.1. Upper end: assumes every critical error interrupts its node (paper's footnote 14). | 1–15 /GPU-y | Cui et al. 2025 [V, derived] |
| MTTR | 72 h | **Replaced:** expected node service time 0.88 h (A100) and 2.2 h (H100); individual incidents range from 2 min to 23 node-hours. 72 h was far too pessimistic for the dominant (soft) errors. | 1 min – 72 h | Cui et al. 2025 Sec 5.4 [V] |
| Unrepairable failure share | 0.3 | ASSUMPTION | 0.1–0.6 | |
| Replacement part embodied share | 0.5 | ASSUMPTION | 0.2–1.0 | |
| Pool size | 64 GPUs | ASSUMPTION | 8–512 | Deployment granularity |
| Availability target | 99.9% | ASSUMPTION | 99–99.99% | SLO practice |
| Independent failures | yes | ASSUMPTION (known false) | beta-binomial correlation | Rack/power-domain correlation is planned |
| Power-cap curve (throughput, power, temperature vs cap) | placeholder `PLACEHOLDER_CURVE` | **Evidence says caps rarely bind in decode.** On an H200 with vLLM, memory-bound decode draws 147–292 W of a 700 W rating, and throughput moves <3% across 280–700 W caps. Caps engage only for large batch × long generation (up to 44% throughput loss at 280 W). **SM clock locking** saves 20–32% of decode energy at ≤1% throughput loss (780 MHz, ≤4K output tokens). So the lever to model is clock/DVFS, not the cap. | clock 0.4–1.0 of max | Ma et al., PPAM 2026 [V]; our G1 harness can sweep clocks too |
| Spare idle power | 0.10 kW | ASSUMPTION → **measure with G3 idle** | | `gpu_bench` |
| Software-failure repair time | part of MTTR | ASSUMPTION → **measure with G3 coldstart/fault** | | `gpu_bench` |
| Failure blast radius (pool granularity) | 64 GPUs, independent | ASSUMPTION → **G2**: one failure takes out a whole TP group | TP 1–8 | `gpu_bench` |
| Arrhenius activation energy Ea | 0, 0.3, 0.7 eV | ASSUMPTION (cannot be measured in hours) | 0–1.0 eV; 0 = no thermal aging | Reliability-physics literature (find and verify sources) |

## Fleet-design and uncertainty experiments (`configs/fleet.yaml`)

| Parameter | Value | Status | Notes |
|---|---|---|---|
| Per-GPU efficiency vs tensor-parallel degree | TP1 1.0, TP2 0.92, TP4 0.85, TP8 0.75 | ASSUMPTION | Replaced by a break-even analysis: we report the efficiency gain a bigger group *needs*, not which group wins |
| KV cache + activations share of weights | 0.3 | ASSUMPTION | Sets minimum TP (70B fp16 on 80 GB GPUs → TP ≥ 4) |
| Decode throughput per GPU at TP1 | 1000 tok/s | ASSUMPTION | Cancels in all relative results |
| Hardware amortization lifetime | 4 years | Matches EcoServe's baseline (Sec 6.5) [V] | EcoServe's Recycle uses 9 y hosts / 3 y GPUs |
| Clock-lock curve | 780 MHz: 0.99 throughput, 0.74 power; 1590 MHz: ~1.0 throughput, 0.92 power | Sourced (memory-bound decode, H200) | Ma et al. PPAM 2026; other GPU generations may differ |
| Error-rate uncertainty | triangular (3.75, 7.5, 15) per GPU-year | Mode sourced (Delta H100), range ASSUMPTION (x0.5–x2) | |
| Recovery-time uncertainty | triangular (5 min, 2.2 h, 8 h) | Range informed by Delta incidents (2 min – 23 node-hours); shape ASSUMPTION | 2.2 h is Delta's *expected* H100 service time |
| Embodied-carbon uncertainty | triangular (800, 1000, 1600) kg | Right-tail shape from CarbonClarity (mean-to-p95 up to 1.6x); level ASSUMPTION | |
| Grid-intensity forecast error | x(0.7, 1.0, 1.3) | ASSUMPTION | |
