# Related work: verified notes

Status legend: **[V]** checked against the arXiv abstract or HTML; **[PDF]** still to be
verified against the full paper before citing.

## EcoServe: Designing Carbon-Aware AI Inference Systems
Y. Li, Z. Hu, E. Choukse, R. Fonseca, G. E. Suh, U. Gupta. arXiv 2502.05043 (v2, Mar 2025). [V]
* Abstract: up to 47% lower carbon vs performance-, energy- and cost-optimized design
  points while meeting SLOs. [V]
* 4R pillars: Reuse (offline inference on idle host CPUs), Rightsize (GPU type/count),
  Reduce (trim host DRAM/SSD), Recycle (asymmetric host/GPU lifetimes). [V for definitions; PDF for numbers]
* Per-task carbon: (P_host + P_gpu)·t·CI + (CF_host_emb + CF_gpu_emb)·t/LT, i.e. linear
  amortization over lifetime LT. [V]
* ILP for provisioning; Recycle reportedly hosts about 9 y, GPUs about 3 y, about 16%
  saving over 10 y. [PDF]
* No failure, spare-capacity or availability model found in the part read so far. [PDF: confirm]

## CarbonClarity: Understanding and Addressing Uncertainty in Embodied Carbon for Sustainable Computing
X. Chen, L. Han, A. Bhagavathula, U. Gupta. arXiv 2507.01145 (v2, Jan 2026); linked DOI
suggests ICCAD 2025. [V]
* Embodied carbon as distributions (energy/area, gas/area, yield, fab carbon intensity);
  mean-to-p95 gap up to 1.6x at 7 nm; chiplets cut p95 by 18% for a mobile case. [V]

## Story of Two GPUs: Characterizing the Resilience of Hopper H100 and Ampere A100 GPUs
S. Cui, A. Patke, H. Nguyen, A. Ranjan, Z. Chen, P. Cao, G. Bauer, B. Bode, C. Di Martino,
S. Jha, C. Narayanaswami, D. Sow, Z. T. Kalbarczyk, R. K. Iyer. SC '25. arXiv 2503.11901. [V]
* NCSA Delta: 448 A100 (895 days) + 608 H100/GH200 (146 days), ~11.7M GPU-hours.
* Per-GPU MTBE for uncorrectable ECC: 283,271 h (A100) vs 88,768 h (H100), i.e. 3.2x lower on H100.
* Hazard rate "exhibits no clear temporal trends during the operational period" (Nelson-Aalen;
  calendar time, not GPU age). Infant-mortality phase is visible (pre-operational MTBE 0.15 h).
* Node availability ~99.4% (A100) and ~99.3% (H100); expected node service 0.88 h / 2.2 h.
* Their simulation: a 608-GPU job needs ~5% overprovisioning for 99.9% availability at 2.2 h
  recovery, ~2% at 5 min. **Our pool model gives 3.9% / 0.8% for 4-GPU failure units**, the
  same order (different availability definition).
* Replacements: 4 A100 (boot failure) in 895 d; 2 H100 (memory) in 146 d.

## The Illusion of Power Capping in LLM Decode
B. Ma, A. Afzal, J. Eitzinger, G. Wellein. PPAM 2026. arXiv 2605.11999. [V]
* One H200 (700 W), vLLM eager, ~4B models of five attention architectures.
* Decode draws 147–292 W; throughput moves <3% across 280–700 W caps; caps bind only at large
  batch × long generation (up to 44% throughput loss at 280 W for attention models).
* SM clock locking: 20–32% decode-energy saving at ≤1% throughput loss (≤4K output tokens).
* Limitations: single GPU, single framework, dense models, eager mode.

## Revisiting Reliability in Large-Scale Machine Learning Research Clusters
A. Kokolis, M. Kuchnik, et al. (Meta FAIR). HPCA 2025. arXiv 2410.21680. [V: abstract only]
* 11 months, 4M jobs, >150M A100 GPU-hours on two clusters (16k and 8k GPUs).
* Large jobs most failure-exposed. Specific MTTF numbers: verify from the paper before citing.

## To add (verify each before citing)
POLCA / Patel et al., "Characterizing Power Management Opportunities for LLMs in the Cloud",
ASPLOS 2024 (PDF host unreachable from this environment; not yet read).
ACT (ISCA 2022); Fair-CO2 (ISCA 2025); CORDOBA (HPCA 2025); Carbon Explorer (ASPLOS 2023);
Splitwise; Melange; Let's Wait Awhile; GreenSKU; public large-cluster failure studies;
fuzzy multi-objective MILP (Poonia et al., Management of Environmental Quality 35(7), 2024).
