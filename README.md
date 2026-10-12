# How brand awareness moves CAC, and why country size is a red herring

A hierarchical Bayesian marketing mix model with awareness mediation, built to answer
an international expansion question: **if we spend to build awareness in a new country,
what happens to CAC — and does the answer depend on how big the country is?**

**→ [Read the write-up](https://duanchenliu.github.io/oura-awareness-cac/)**
· [Analysis notebook](notebooks/awareness_cac_analysis.ipynb)

---

> ### ⚠️ All data in this project is simulated
> No company-internal data was used or available. I built a generative model of 18
> international markets whose *structural anchors* match figures Oura disclosed publicly
> in its Form S-1 (filed 2026-09-03): ~40% of new subscribers arriving organically,
> ~85% twelve-month paid retention, international markets under 20% of hardware sales.
> Everything else is invented and documented in [`src/simulate.py`](src/simulate.py).
>
> Simulating is the point: because I know the true parameters, I can test whether the
> modelling approach recovers them. It does not, and that turned out to be the most
> useful finding.

## Findings

| Finding | Evidence |
|---|---|
| **Population washes out.** At matched awareness, awareness-driven CAC is set by qualified buyer density and media cost, not market size. | log-population coefficient −0.11 (t = −0.66), R² = 0.95. Bivariate correlation is a misleading +0.91. |
| **An observational MMM understates the awareness effect by ~69%**, with a credible interval that misses the truth. | true γ = 1.81 vs estimated γ = 0.56 |
| **The bias has a price.** It recommends 17% of budget to brand when the right answer is 44%. | 14.8% fewer activations at identical spend |
| **Partial pooling is what makes new-market forecasting work.** | CAC forecast MAPE 71% → 19% → 7% |
| **Awareness and acquisition need different carryover rates.** One adstock for both flips the sign of the brand coefficient. | θ = 0.95 vs θ = 0.30 |

## The model

**Stage 1 — awareness response**

```
logit(A_ct) = a_c + b_c · log(1 + adstock(brand_ct) / (pop_c · cpm_c))
```

**Stage 2 — acquisition response**

```
log(activations_ct / TAM_c) = α_c + β_c·log(1 + perf̃_ct) + γ_c·A_ct
                            + δ·doors + ζ·penetration + season + ε
```

**Partial pooling**

```
α_c ~ N(μ_α, τ_α)    β_c ~ N(μ_β, τ_β)    γ_c ~ N(μ_γ + κ·z_c, τ_γ)
```

`κ` tests whether awareness is more potent where qualified buyers are dense. It came
back null (0.06 ± 0.17) — awareness works about equally well everywhere, so all the
cross-country variation lives in the *price* of awareness and the *size of the
qualified base*.

## Why population cancels

```
                     pop × cpm                 cpm
CAC_awareness  ∝  ─────────────────────  =  ─────────────
                  pop × density × γ         density × γ
```

Large markets cost more to make aware **and** contain proportionally more buyers. The
two effects offset. What does not cancel is how concentrated the qualified buyers are.

## Reproduce

```bash
pip install -r requirements.txt
python3 src/simulate.py     # generate the 18-market panel
python3 src/analysis.py     # fit, analyse, write figures + results.json
python3 src/build_notebook.py && jupyter nbconvert --execute --inplace \
    notebooks/awareness_cac_analysis.ipynb
```

Full run is about four minutes. `REFIT=1` forces a resample of the cached main posterior.

## Layout

```
src/simulate.py        generative model of 18 markets (the data generating process)
src/model.py           feature construction, hierarchical PyMC models, response curves
src/analysis.py        every analysis + figure in the study
src/build_notebook.py  assembles the narrative notebook
notebooks/             executed analysis notebook
figures/               all charts
data/                  simulated panel, ground truth, posteriors, results.json
index.html             the write-up
```

## Limitations

- All data is simulated; real data could behave differently, most plausibly if awareness
  effects vary by market in ways the simulator holds constant.
- Carryover rates are grid-searched rather than jointly estimated, so that uncertainty is
  not propagated into the posterior.
- The geo experiment is simulated at a precision a real test would need meaningful spend
  and several weeks to reach.
- Qualified density is invented. In production it is a research deliverable, and the whole
  result is sensitive to it.

---

Duanchen Liu · [duanchenliu.github.io](https://duanchenliu.github.io) · October 2026
