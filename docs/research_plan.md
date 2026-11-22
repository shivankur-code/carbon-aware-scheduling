# Research plan

**Working title:** *Does keeping AI hardware longer actually save carbon? Reliability- and
uncertainty-aware lifetime planning for LLM inference fleets*

**One-sentence pitch:** Carbon-aware inference design says to keep hardware longer on clean
grids. We show what happens to that advice once hardware wears out, spares cost embodied
carbon, and every input (embodied carbon, grid intensity, failure rates) is uncertain. We
then show that **one operational knob, the GPU power cap, acts on all four carbon terms at
once**, with the effects measured on real GPUs.

## Evidence update (2026-11-22): what published data changed

Checking our assumptions against field data overturned two of them and sharpened the thesis.

| Assumption | What the evidence says | Consequence |
|---|---|---|
| GPUs wear out (Weibull k = 2), so old GPUs need more spares | Delta (Cui et al., SC'25): **no clear temporal trend in hazard rate** over 2.5 years of A100 operation; hard replacements are rare (~0.4–0.8 %/year). Newer H100s have **3.2x lower** per-GPU MTBE for uncorrectable memory errors than A100s. | Wear-out is not the main reliability cost in years 0–2.5, and "newer = more reliable" is false. Keep wear-out only as a swept scenario for ages beyond the data. |
| Repairs take days (MTTR 72 h) | The dominant errors are soft and fixed by a reset/reboot: expected node service **0.88 h (A100), 2.2 h (H100)**; ~13.5 / 7.5 critical errors per GPU-year. | Spare capacity is driven by **error rate × recovery time × blast radius**, not by hardware age. |
| Power caps are a strong lever | Ma et al. (PPAM 2026): LLM decode draws **147–292 W of 700 W**, so caps rarely bind; **clock locking** saves 20–32% decode energy at ≤1% throughput loss. | Model the lever as clock/DVFS. Power caps matter only for long, large-batch generation. |

### Sharpened thesis

> **The embodied-carbon cost of reliability in LLM serving is set by how fast you recover and
> how much a single error takes down, not by how old the hardware is.**

`experiments/recovery_spares.py` (`docs/figures/recovery_spares.png`), using Delta's error rates:
to keep a 512-GPU fleet at 99.9% availability,

| Recovery time | Blast radius 1 GPU | 4-GPU node | 8-GPU tensor-parallel group |
|---|---|---|---|
| 5 min | ~0.2% extra GPUs | 0.6–1.6% | 1.6–3.1% |
| 0.88–2.2 h (Delta's measured service times) | 0.6–1.4% | 2.3–5.5% | 4.7–10.9% |
| 1 day | 4–7% | 17–27% | 36–56% |

(Ranges span H100 and A100 error rates. Upper-end rates: every critical error is assumed to
interrupt its node, as Delta's own availability analysis assumes.)

Spare GPUs carry full embodied carbon, so **recovery engineering (fast restart, hot spares,
smaller failure domains) is a carbon lever**, and it interacts with tensor-parallel degree
(bigger groups → better throughput, bigger blast radius). This builds directly on the
operator-side expertise the project is meant to showcase (fault tolerance, node lifecycle).

### Results so far (sourced reliability inputs; remaining assumptions in `docs/assumptions.md`)

**R1. Recovery time × blast radius sets the spare overhead** (`experiments/recovery_spares.py`).
See the table above. At Delta's measured service times (0.9–2.2 h) the overhead is modest
(about 1–5% of a 500-GPU fleet). It becomes large only at day-scale recovery or with 8-GPU
failure domains.

**R2. Fleet design** (`experiments/fleet_design.py`, `docs/figures/fleet_design.png`), 70B model,
~500 H100-class GPUs:
* **Clock locking** (Ma et al. curve) cuts total annual carbon by **8% (low-carbon grid), 20%
  (mid) and 23% (high)**, the largest single lever in the model.
* **Spares are a small slice** of total carbon at measured recovery times (≈4% of GPUs).
* **Blast-radius break-even:** an 8-GPU replica must be **2–4% more efficient per GPU** than a
  4-GPU replica to break even at hour-scale recovery, rising to **16–32% at 1–3 days on a
  low-carbon grid** (8.5–16% mid, 6.5–14% high). Reliability matters most where embodied carbon
  dominates: clean grids.

**R3. Promised vs realized under uncertainty** (`experiments/uncertainty_planning.py`,
`docs/figures/uncertainty_planning.png`), 70B fleet, mid grid, 4,000 scenarios:
* The **deterministic plan** (spares sized at most-likely error rate and recovery time)
  promises 99.96% availability but **misses the 99.9% target in 68% of scenarios**; its
  worst-5% availability is 87%.
* A **fuzzy alpha = 0.5 plan** (same spare count as a 95% scenario chance constraint) misses in
  **4.7%** of scenarios for **+1.9% annual carbon**; alpha = 0.25 misses in 0.05% for +3.1%.
* Mean realized carbon is ~3% above the promised value for every plan because of the
  embodied-carbon right tail.

**Candidate headline for the paper/email (all numbers reproducible from the repo):**
> "Using field reliability data from 1,056 A100/H100 GPUs, we find the carbon cost of keeping
> an LLM inference fleet available is set by recovery time and failure blast radius, not
> hardware age. A plan sized at most-likely failure rates misses its 99.9% availability
> target in 68% of plausible scenarios; a fuzzy (alpha = 0.5) plan fixes this for under 2%
> extra carbon, while clock locking saves 8–23%."

### Revised research questions (supersede §3 where they differ)
* **RQ1′:** How much embodied carbon does reliability cost in an LLM inference fleet, as a
  function of error rate, recovery time and blast radius (tensor-parallel group size)?
* **RQ2′:** When does lifetime extension (EcoServe's Recycle) stay carbon-positive, given
  that observed hazard is flat early but unknown later? Treat late-life hazard as deep
  uncertainty and report the break-even late-life failure rate.
* **RQ3′ (uncertainty, unchanged):** promised vs realized savings under uncertain embodied
  carbon, grid intensity and error rates (fuzzy / scenario methods).
* **RQ4′:** Joint provisioning: tensor-parallel degree × spares × clock level × lifetime.

### What the GPU runs are now for
Optional validation, not the foundation: (a) measured **restart / recovery time** of a vLLM
server (G3) directly sets the x-axis of the headline figure; (b) a **clock-lock sweep**
(replacing the power-cap sweep) to confirm Ma et al. on another GPU generation.

## 0. The core idea: one knob, four carbon effects

| Effect of lowering the power cap | Carbon term | Measured or modeled |
|---|---|---|
| Lower power, lower energy per token (if throughput drops less than power) | Operational ↓ | **Measured** (G1) |
| Lower throughput, so more GPUs are needed | Embodied ↑ | **Measured** (G1) |
| Lower temperature, slower wear-out, longer viable lifetime | Embodied ↓ (amortized), spares ↓ | Temperature **measured**; temperature-to-aging via Arrhenius (Ea swept from 0) |
| Freed power headroom the grid can call on | Grid flexibility | Modeled (RQ5) |

EcoServe chooses *which* hardware to provision and *how long* to keep it. This work adds
*how hard to run it*, and shows that this choice cannot be separated from lifetime: the
carbon-optimal (cap, lifetime) pair depends on the grid. Pilot (placeholder curve,
`docs/figures/derating.png`): on dirty grids the cap pays for itself through energy
efficiency; on clean grids it pays off mainly by extending lifetime, and only if thermal
aging is real (Ea > 0). The measured curve decides how large each regime is.

---

## 1. The gap we are pushing on

| Prior work | What it does | What it leaves open (our entry point) |
|---|---|---|
| **EcoServe** (Li et al., arXiv 2502.05043, 2025) | Four pillars for carbon-aware LLM inference. *Reuse* runs offline work on idle host CPUs. *Rightsize* chooses GPU type and count with an ILP. *Reduce* trims host DRAM and SSD. *Recycle* uses **asymmetric lifetimes**: hosts kept longer (about 9 y), GPUs replaced sooner (about 3 y). Embodied carbon is amortized linearly over lifetime. | The lifetimes are chosen **deterministically**. We found no failure, spare-capacity or availability model in the part of the paper read so far, and grid intensity is handled as fixed sensitivity points, not a distribution. **(Verify against the PDF before stating this publicly.)** |
| **CarbonClarity** (Chen et al., arXiv 2507.01145, 2025) | Embodied carbon as a probability distribution; mean-to-p95 gap up to 1.6x at 7 nm. | The uncertainty is not carried into *fleet operating decisions* such as lifetime, spares or provisioning. |
| ACT, Fair-CO2, CORDOBA, Carbon Explorer (Gupta et al.) | Carbon modeling, attribution and optimization frameworks | Hardware treated as always available |

**The open question:** "Recycle" (extending lifetime) is the pillar most exposed to hardware
aging, yet aging is the one effect it does not model. Older hardware fails more often. Each
failure needs spare capacity, and spares carry their own embodied carbon and idle power.
Every lifetime extension also has to meet the same SLO. Nobody has quantified how much of
the lifetime-extension carbon saving survives once these effects are included, or how
uncertain that answer is.

## 2. Pilot evidence (placeholder parameters, not a finding)

`python experiments/pilot_lifetime.py` produces `docs/figures/pilot_lifetime.png` from an
analytic steady-state fleet model with Weibull wear-out and binomial spare sizing
(`src/calf/models/`). Under **assumed** parameters:

* On dirty grids (above roughly 150 gCO2e/kWh) the carbon-optimal GPU lifetime is short
  (2–4 y), and failures barely move it.
* On clean grids (below roughly 50 g/kWh) a failure-oblivious planner keeps GPUs about
  8–10 y. A failure-aware planner picks a shorter lifetime, and the oblivious plan costs up
  to about 13% extra carbon once the spares it needs are added.
* **The bigger effect is availability.** Without spares, the oblivious plan's pools are
  fully available only 65–93% of the time on clean grids (target: 99.9%).

So the interesting regime is exactly the one carbon-aware design pushes toward: **clean
grids plus long-lived hardware**. Hosts, at about 9 years in EcoServe's Recycle, are the
first case to test.

## 3. Research questions and hypotheses

| # | Question | Hypothesis (to test, not assume) |
|---|---|---|
| RQ1 | How do wear-out failures, repair time and maintenance drains change the carbon-optimal lifetime of GPUs and of hosts? | H1: Failure-aware lifetimes are noticeably shorter only on low-carbon grids. On those grids, the cost of ignoring failures shows up mainly as missed SLOs, not extra carbon. |
| RQ2 | Do asymmetric lifetimes (long-lived hosts, short-lived GPUs) stay carbon-optimal once a 9-year host's failure rates and spares are included? | H2: Asymmetric lifetimes still win, but the best host lifetime shrinks and depends on the grid. The saving is smaller than the deterministic estimate. |
| RQ3 | With embodied carbon (CarbonClarity-style distributions), grid-intensity forecasts and failure rates all uncertain, how much of the promised saving is realized? Which planner is most robust? | H3: Deterministic plans over-promise on clean grids. A fuzzy (alpha-cut) or scenario-robust plan gives up a little expected carbon for much lower worst-case regret. |
| RQ4 | Integrated fleet planning: which GPU types, how many, how many spares, at which ages, in which regions, at what lifetime? | H4: Reliability-aware provisioning (failures plus uncertainty) traces a carbon–cost–availability Pareto front that dominates the EcoServe-style deterministic plan once evaluated under realistic failures. |
| RQ5 (stretch) | Can spare and aging capacity double as grid-flexibility headroom (demand response), or does it conflict with SLO reserves? | Open |

## 4. Contributions (target)

1. **Reliability-adjusted carbon model** for inference fleets: age-dependent hazard, repair
   and drain unavailability, k-out-of-n spare sizing, and the embodied and idle carbon of
   spares. Extensions: correlated (rack- or power-domain) failures and repair time that
   grows with age (parts become scarce).
2. **Reliability-aware Recycle:** carbon-optimal GPU and host replacement ages as a
   function of grid intensity and failure regime, shown as a phase diagram. Compared
   against deterministic asymmetric-lifetime planning.
3. **Uncertainty-aware planning:** triangular fuzzy numbers solved with alpha-cuts
   (following the multi-objective fuzzy MILP approach), plus scenario and chance-constrained
   versions. Reported as promised vs realized savings, value of the stochastic solution,
   and the probability that lifetime extension saves carbon.
4. **Open, reproducible tool** (`calf`): every figure comes from one config and one
   command, using public data only.

## 5. Method

### 5.1 Models (`src/calf/models/`)
* `reliability.py` (done, pilot): Weibull hazard, unavailability, binomial spare sizing.
* `lifetime.py` (done, pilot): steady-state staggered fleet; throughput per watt doubles
  every T years; embodied carbon amortized over L; spares and replacement parts.
* Next: separate **host** and **GPU** lifetimes (EcoServe-style asymmetric), where host
  failures take out all GPUs on the server. Correlated failure domains via a beta-binomial
  model. Age-dependent MTTR.
* Next: a *transient* discrete-time simulator (`simulate/`) driven by an hourly
  carbon-intensity trace and a diurnal request trace, to validate the analytic
  steady-state model.

### 5.2 Optimization (`src/calf/optimize/`)
* Provisioning MILP (PuLP + HiGHS): integer counts per (GPU type, age cohort, region),
  spares, and lifetime chosen from a discrete menu; SLO capacity constraints; objectives:
  carbon, cost, availability.
* Pareto fronts with the **epsilon-constraint** method.
* Uncertainty variants: fuzzy (alpha-cuts), scenario-based stochastic (SAA), and
  chance-constrained availability.

### 5.3 Baselines
1. Performance-only (newest GPUs, sized for peak, 4-year symmetric lifetime).
2. Cost-only.
3. Carbon-greedy, failure-oblivious.
4. **EcoServe-inspired deterministic plan** (simplified reimplementation, labeled as
   "inspired by", not the official system): asymmetric lifetimes chosen without failures.
5. This work: failure-aware, plus uncertainty-aware.

## 6. Experiments → figures (each: one config, one command)

| ID | Figure | Answers |
|---|---|---|
| E1 | Phase diagram: optimal GPU lifetime over (grid intensity × failure regime) | RQ1 |
| E2 | Host vs GPU lifetime: carbon surface; deterministic vs failure-aware optimum | RQ2 |
| E3 | Promised vs realized savings under uncertainty, per planner | RQ3 |
| E4 | Carbon–availability Pareto fronts, three grids (low/mid/high) | RQ4 |
| E5 | Sensitivity tornado: embodied carbon, Weibull shape/scale, MTTR, doubling period, PUE | Robustness |
| E6 | Model validation: analytic vs simulated spares and availability | Credibility |
| E7 (stretch) | Dependable demand-response relief vs SLO | RQ5 |

## 6b. GPU measurement campaign (`gpu_bench/`)

Each campaign replaces model assumptions with measurements. One command per campaign,
results committed to `data/measurements/gpu/<gpu>_<date>/`.

| ID | Config | Measures | Replaces assumption | GPU need | Est. time |
|---|---|---|---|---|---|
| G0 | `g0_smoke.yaml --quick` | Harness works, power capping permitted | none | any 1 GPU | 10 min |
| G1 | `g1_powercap.yaml` | Throughput, power, temperature, SM clock, TTFT/TPOT, SLO attainment, J/token vs power cap (50–100%) at 4 load levels | Cap curve (tau, rho, T), active power | 1 GPU; **repeat on each generation available** (e.g. T4/V100, A10/A100, L4/L40S, H100) | about 3 h per GPU type |
| G1x | G1 on older vs newer GPUs | perf/W across generations; how much efficiency an older GPU gives up | Doubling period; "keep old GPU vs buy new" | as many types as possible | about 3 h each |
| G2 | `g2_tp_blast_radius.yaml` | Efficiency vs tensor-parallel degree; TP degree = failure blast radius | Pool size and failure domain in the spare model | 4–8 GPU node | about 3 h |
| G3 | `g3_reliability.yaml` | Idle power (no process / model loaded), cold-start time, kill-and-restart downtime and lost requests under load | Spare idle power, software-failure MTTR | 1 GPU | about 30 min |

**What makes this new:** EcoServe and others profile performance and power. As far as we
know (to verify), no study links measured power-cap curves, temperature, failure blast
radius and recovery time to a *lifetime and spares* carbon model.

Measurement integrity: record GPU model, driver, vLLM version, cooling and ambient where
known (`env.json`); repeat at least one configuration to estimate run-to-run noise;
report GPU-only energy, adding host energy as a modeled term.

## 7. Data (public only; record source and licence for each in `data/README.md`)

* **Grid carbon intensity:** Electricity Maps open historical data, UK Carbon Intensity
  API, ENTSO-E, US EIA hourly grid data. Choose three regions: low, mid and high carbon.
* **Requests:** the public Azure LLM inference traces, BurstGPT, or synthetic diurnal traces.
* **GPU power and performance:** vendor TDP specs, MLPerf Inference results.
* **Embodied carbon:** ACT parameters, vendor product carbon footprints, CarbonClarity ranges.
* **Failures (the hardest input):** public large-cluster reliability reports, e.g. GPU
  interruption counts reported in large-model training papers and published
  cluster-reliability studies, plus classic DRAM and disk field studies. **Find and verify
  each source before using it.** When a parameter has no public source, label it ASSUMPTION
  and sweep it. Never use non-public operator data.

## 8. Milestones

| Week | Output | Done when |
|---|---|---|
| 0 (now) | Repo scaffold, pilot model, pilot figure, this plan | Tests pass; pilot reproduces from one command |
| 1 (GPU) | G0 smoke + G3 + G1 on the first GPU type | Measured cap curve replaces the placeholder in `derating.png` |
| 1 | Verify EcoServe/CarbonClarity details from the PDFs (`docs/related_work.md`); collect failure-rate sources; host+GPU asymmetric model | Every parameter in `docs/assumptions.md` has a source or ASSUMPTION tag |
| 2 | E1, E2, E5 with sourced parameters | Phase diagram + host/GPU figure |
| 3 | Uncertainty module (fuzzy + scenario) → E3 | Promised vs realized numbers |
| 4 | Provisioning MILP + baselines → E4; simulator → E6 | Pareto plot from one command |
| 5 | Paper draft via paper-orchestra (`paper/`); 2-page note | One honest headline number backed by the repo |
| 6+ | Stretch E7; workshop submission (e.g. HotCarbon) | Optional |

**Ready to send to Prof. Gupta when:** E1–E5 exist with sourced parameters, README shows
the figures, and the note states one headline finding with honest caveats.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Effect is small with real failure rates | Report it honestly. "Recycle is robust to failures" is also a useful result. Pivot the headline to the availability cost and to host lifetime. |
| No public age-dependent GPU failure data | Treat failure regime as a swept axis; show where conclusions flip (break-even maps). |
| Misrepresenting EcoServe | Read the full PDF; quote only verified numbers; call our baseline "EcoServe-inspired". |
| Scope creep | RQ1–RQ3 are the paper; RQ4 is engineering polish; RQ5 is optional. |

## 10. Paper hand-off (paper-orchestra)

`paper/outline.md` maps each section to the experiments and figures above.
`paper/references.bib` holds verified citations only. Once paper-orchestra's expected input
format is known, adapt `paper/` to it.
