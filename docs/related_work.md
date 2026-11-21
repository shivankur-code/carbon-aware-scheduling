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

## To add (verify each before citing)
ACT (ISCA 2022); Fair-CO2 (ISCA 2025); CORDOBA (HPCA 2025); Carbon Explorer (ASPLOS 2023);
Splitwise; Melange; Let's Wait Awhile; GreenSKU; public large-cluster failure studies;
fuzzy multi-objective MILP (Poonia et al., Management of Environmental Quality 35(7), 2024).
