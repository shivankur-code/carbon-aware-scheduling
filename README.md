# calf — reliability- and uncertainty-aware carbon planning for LLM inference fleets

> **Status: early research prototype (pilot).** Numbers in this repo come from
> placeholder parameters until `docs/assumptions.md` says otherwise. Do not cite them.

Carbon-aware inference design (e.g. EcoServe) says to keep hardware longer on clean grids so
its embodied carbon is spread over more years. Older hardware also fails more often, needs
spare capacity, and must still meet latency SLOs. **This project asks how much of that
"keep it longer" saving survives realistic failures and uncertainty, and how operators
should plan GPU and host lifetimes, spares and fleet mix as a result.**

![pilot](docs/figures/pilot_lifetime.png)

*Pilot (assumed parameters): failures barely matter on dirty grids, where hardware is
replaced quickly anyway. On clean grids a failure-oblivious plan keeps GPUs about 8–10
years and misses its availability target badly unless spares are added.*

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q
python experiments/pilot_lifetime.py      # -> docs/figures/pilot_lifetime.png
```

## Layout

```
configs/        experiment configs (YAML, one per figure)
src/calf/
  models/       reliability.py (hazard, spares), lifetime.py (fleet carbon vs lifetime)
  optimize/     provisioning MILP, epsilon-constraint Pareto, fuzzy/scenario (planned)
  simulate/     discrete-time fleet simulator (planned)
  data/         carbon-intensity, workload, hardware loaders (planned)
experiments/    one script per figure
docs/           research_plan.md, related_work.md, assumptions.md, figures/
paper/          outline + references for paper generation
```

## Read next

* [`docs/research_plan.md`](docs/research_plan.md): the research gap, questions, experiments and milestones
* [`docs/assumptions.md`](docs/assumptions.md): every parameter, with source or ASSUMPTION tag
* [`docs/related_work.md`](docs/related_work.md): verified notes on prior work

## Attribution

This project is **inspired by** and builds on published work by Udit Gupta's group and
collaborators, in particular EcoServe (arXiv 2502.05043) and CarbonClarity
(arXiv 2507.01145). It is independent and not affiliated with those authors. It uses
public data only.

## License

TBD (MIT or Apache-2.0).
