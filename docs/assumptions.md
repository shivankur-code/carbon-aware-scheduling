# Assumptions register

Every model parameter must have a **source** or be tagged **ASSUMPTION**, with the range
used in sensitivity analysis. The pilot values are all placeholders.

| Parameter | Pilot value | Status | Sweep range | Source to find / notes |
|---|---|---|---|---|
| Embodied carbon per GPU incl. server share | 1000 kgCO2e | ASSUMPTION | 150–2500 | ACT; vendor product carbon footprints; CarbonClarity distributions. EcoServe's own per-GPU/host values: verify from PDF |
| Average active power per GPU (incl. server share) | 0.40 kW | ASSUMPTION | 0.25–0.9 | Vendor TDP; MLPerf Inference power results |
| Spare idle power | 0.10 kW | ASSUMPTION | 0–0.2 | Spares may be powered off (cold), which trades availability against energy |
| PUE | 1.2 | ASSUMPTION | 1.1–1.5 | Public hyperscaler sustainability reports |
| Perf/W doubling period | 3.5 y | ASSUMPTION (EcoServe reportedly uses 3.5 y; verify) | 2–5 y | Verify in EcoServe PDF; MLPerf generations |
| Weibull shape k | 2.0 | ASSUMPTION | 1 (no wear-out) – 3 | Public large-cluster reliability studies |
| Weibull scale | 5/8/12 y | ASSUMPTION | 4–15 y | As above |
| MTTR | 72 h | ASSUMPTION | 24 h – 30 d | Should grow with age (parts scarcity) |
| Unrepairable failure share | 0.3 | ASSUMPTION | 0.1–0.6 | |
| Replacement part embodied share | 0.5 | ASSUMPTION | 0.2–1.0 | |
| Pool size | 64 GPUs | ASSUMPTION | 8–512 | Deployment granularity |
| Availability target | 99.9% | ASSUMPTION | 99–99.99% | SLO practice |
| Independent failures | yes | ASSUMPTION (known false) | beta-binomial correlation | Rack/power-domain correlation is planned |
| Power-cap curve (throughput, power, temperature vs cap) | placeholder `PLACEHOLDER_CURVE` | ASSUMPTION → **measure with G1** | caps 50–100% | `gpu_bench` |
| Spare idle power | 0.10 kW | ASSUMPTION → **measure with G3 idle** | | `gpu_bench` |
| Software-failure repair time | part of MTTR | ASSUMPTION → **measure with G3 coldstart/fault** | | `gpu_bench` |
| Failure blast radius (pool granularity) | 64 GPUs, independent | ASSUMPTION → **G2**: one failure takes out a whole TP group | TP 1–8 | `gpu_bench` |
| Arrhenius activation energy Ea | 0, 0.3, 0.7 eV | ASSUMPTION (cannot be measured in hours) | 0–1.0 eV; 0 = no thermal aging | Reliability-physics literature (find and verify sources) |
