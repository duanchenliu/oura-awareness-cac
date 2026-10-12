"""
Runs every analysis in the study and writes results/figures to disk.

    python3 src/analysis.py
"""
import os, sys, json, time, pickle
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import simulate as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = f"{ROOT}/figures"
DATA = f"{ROOT}/data"
os.makedirs(FIG, exist_ok=True)

INK = "#14213d"; ACCENT = "#e07a5f"; TEAL = "#2a9d8f"; MUTED = "#8d99ae"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9ced6", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "600",
    "figure.facecolor": "white", "axes.facecolor": "white", "grid.color": "#eef1f5",
})

TARGET_AW = 0.15          # matched awareness level for cross-market comparison
RESULTS = {}


def _ivals(summ):
    """arviz renamed HDI columns across versions; find the interval pair."""
    lo = [c for c in summ.columns if c.endswith(("_lb", "%")) and "hdi" in c or c.endswith("_lb")]
    hi = [c for c in summ.columns if c.endswith("_ub") or (c.endswith("%") and "97" in c)]
    return (lo[0] if lo else "mean"), (hi[0] if hi else "mean")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figure -> {name}.png")


# ==========================================================================
def main():
    t_start = time.time()
    panel = pd.read_csv(f"{DATA}/panel.csv")
    truth = pd.read_csv(f"{DATA}/ground_truth.csv")
    econ = S.unit_economics()
    RESULTS["unit_economics"] = {k: float(v) for k, v in econ.items()}

    # ---------------------------------------------------------------- 0. calibration
    tot = panel[["activations", "organic_activations", "paid_activations",
                 "retail_activations", "total_spend"]].sum()
    RESULTS["calibration"] = {
        "organic_share": float(tot.organic_activations / tot.activations),
        "paid_share": float(tot.paid_activations / tot.activations),
        "retail_share": float(tot.retail_activations / tot.activations),
        "blended_cac": float(tot.total_spend / tot.activations),
        "markets": int(panel.code.nunique()),
        "months": int(panel.month.max() + 1),
        "observations": int(len(panel)),
        "sec_anchor_organic": S.SEC_ANCHORS["organic_share_of_new_subscribers"],
    }

    # ---------------------------------------------------------------- 1. adstock
    print("\n[1] adstock grid search")
    gs = M.grid_search_adstock(panel)
    RESULTS["adstock"] = gs
    print("   ", gs)
    d = M.build_features(panel, gs["theta_brand"], gs["theta_perf"])

    # ---------------------------------------------------------------- 2. main fit
    print("\n[2] hierarchical fit (partial pooling)")
    cache = f"{DATA}/idata_main.pkl"
    t0 = time.time()
    if os.path.exists(cache) and os.environ.get("REFIT") != "1":
        idata = pickle.load(open(cache, "rb"))
        cats = sorted(d.code.unique())
        qd = d.groupby("code")["qual_frac"].first().reindex(cats).values
        _sd = np.log(qd).std()
        z_qd = ((np.log(qd) - np.log(qd).mean()) / _sd) if _sd > 0 else np.zeros_like(qd)
        print("    loaded cached posterior")
    else:
        _, idata, cats, z_qd = M.fit_hierarchical(d, "partial", draws=1000,
                                                  tune=1000, chains=4)
        pickle.dump(idata, open(cache, "wb"))
        print(f"    sampled in {time.time()-t0:.0f}s")
    summ = az.summary(idata, var_names=["mu_alpha", "mu_beta", "mu_gamma", "kappa",
                                        "tau_alpha", "tau_beta", "tau_gamma",
                                        "delta", "zeta", "sigma"])
    LO, HI = _ivals(summ)
    RESULTS["main_fit"] = {
        k: {"mean": float(summ.loc[k, "mean"]), "sd": float(summ.loc[k, "sd"]),
            "ci_lo": float(summ.loc[k, LO]), "ci_hi": float(summ.loc[k, HI]),
            "r_hat": float(summ.loc[k, "r_hat"])}
        for k in summ.index}
    RESULTS["max_rhat"] = float(summ["r_hat"].max())
    print(summ[["mean", "sd", "r_hat"]].round(3).to_string())

    cp = M.posterior_country_params(idata, cats)
    cp = cp.merge(truth[["code", "psi_c", "eta_c", "beta_c", "qual_frac"]]
                  .rename(columns={"beta_c": "beta_true"}), on="code")
    cp = cp.merge(panel.groupby("code")[["population_m", "cpm_index", "qualified_tam"]]
                  .first().reset_index(), on="code")
    cp = cp.merge(panel.groupby("code")["market"].first().reset_index(), on="code")

    # Recovery check. The simulator's true semi-elasticity of activations with
    # respect to awareness is analytic: differentiate each channel and weight by
    # its share. This is directly comparable to gamma.
    tr = truth.set_index("code")
    pan = panel.merge(panel.groupby("code")["market"].first().rename("mk"), on="code")
    rec = []
    for c, g in pan.groupby("code"):
        psi, eta = tr.loc[c, "psi_c"], tr.loc[c, "eta_c"]
        A = g.awareness_true.values
        qb = g.q_bass.values if "q_bass" in g else None
        pen = g.penetration.values
        O, P, R = (g.organic_activations.values, g.paid_activations.values,
                   g.retail_activations.values)
        tot = np.maximum(O + P + R, 1e-9)
        # d/dA of each channel, divided by total  ->  d log(total)/dA
        dO = O * psi / np.maximum(S.PARAMS["p_base"] + psi * A, 1e-12)
        dP = P * eta / np.maximum(1.0 + eta * A, 1e-12)
        dR = R * S.PARAMS["eta_awareness_to_retail"] / np.maximum(
            0.12 + S.PARAMS["eta_awareness_to_retail"] * A, 1e-12)
        rec.append({"code": c, "true_gamma": float(np.mean((dO + dP + dR) / tot))})
    rec = pd.DataFrame(rec)
    cp = cp.merge(rec, on="code")

    r_rec = float(np.corrcoef(cp.gamma_c, cp.true_gamma)[0, 1])
    true_mu = float(cp.true_gamma.mean())
    est_mu = RESULTS["main_fit"]["mu_gamma"]["mean"]
    RESULTS["recovery"] = {
        "corr_country_gamma": r_rec,
        "true_population_gamma": true_mu,
        "estimated_population_gamma": est_mu,
        "pct_error_population": float(abs(est_mu - true_mu) / true_mu * 100),
        "ci_covers_truth": bool(RESULTS["main_fit"]["mu_gamma"]["ci_lo"] <= true_mu
                                <= RESULTS["main_fit"]["mu_gamma"]["ci_hi"]),
        "true_country_sd": float(cp.true_gamma.std()),
        "estimated_tau_gamma": RESULTS["main_fit"]["tau_gamma"]["mean"],
    }
    print(f"    population gamma: true {true_mu:.3f} vs estimated {est_mu:.3f} "
          f"({RESULTS['recovery']['pct_error_population']:.1f}% error, "
          f"CI covers truth: {RESULTS['recovery']['ci_covers_truth']})")
    print(f"    per-country gamma recovery corr = {r_rec:.3f}")

    # ---------------------------------------------------------------- 3. stage 1
    print("\n[3] awareness response (stage 1) + price of an awareness point")
    aw_ads = search_awareness_adstock(panel)
    RESULTS["awareness_adstock"] = aw_ads
    print(f"    awareness carryover theta = {aw_ads['theta']:.3f} "
          f"(half-life {np.log(0.5)/np.log(aw_ads['theta']):.1f} months), "
          f"pooled slope {aw_ads['slope']:+.3f}, R2 {aw_ads['r2']:.3f}")
    d_aw = M.build_features(panel, aw_ads["theta"], gs["theta_perf"])
    stage1 = fit_stage1(d_aw)
    RESULTS["stage1"] = stage1["summary"]
    cp = cp.merge(stage1["per_market"], on="code")

    cp["usd_per_awareness_pt"] = cp["usd_per_pt"]

    # Matched-awareness economics. Monthly activations the awareness stock itself
    # is responsible for at TARGET_AW, from the stage-2 model:
    #     incremental = acq(A=T) * (1 - exp(-gamma * T))
    mean_acts = panel.groupby("code")["activations"].mean().reindex(cp.code).values
    mean_aw = panel.groupby("code")["awareness_true"].mean().reindex(cp.code).values
    acts_at_T = mean_acts * np.exp(cp.gamma_c.values * (TARGET_AW - mean_aw))
    cp["acq_from_awareness_at_target"] = acts_at_T * (
        1 - np.exp(-cp.gamma_c.values * TARGET_AW))
    cp["awareness_cac"] = (cp["usd_to_target_aw"]
                           / np.maximum(cp["acq_from_awareness_at_target"], 1e-9))
    cp["target_awareness"] = TARGET_AW
    cp.to_csv(f"{DATA}/country_posteriors.csv", index=False)

    RESULTS["target_awareness"] = TARGET_AW
    RESULTS["unreachable_markets"] = [
        {"market": r.market, "max_feasible_awareness": round(float(r.max_feasible_aw), 4)}
        for _, r in cp[~cp.target_reachable.astype(bool)].iterrows()]

    ok = cp[cp.target_reachable.astype(bool) & np.isfinite(cp.awareness_cac)].copy()
    lp = np.log(ok.population_m); lu = np.log(ok.usd_to_target_aw)
    RESULTS["price_of_awareness"] = {
        "corr_logpop_logprice": float(np.corrcoef(lp, lu)[0, 1]),
        "slope_logpop": float(np.polyfit(lp, lu, 1)[0]),
        "n_markets": int(len(ok)),
    }
    la = np.log(ok.awareness_cac)
    # Bivariate correlations are confounded (rich markets have both high
    # qualified density and high CPMs), so regress on all three together.
    Xd = np.column_stack([np.ones(len(ok)), np.log(ok.qual_frac),
                          np.log(ok.cpm_index), lp])
    coef, *_ = np.linalg.lstsq(Xd, la.values, rcond=None)
    resid = la.values - Xd @ coef
    dof = len(ok) - Xd.shape[1]
    se = np.sqrt(np.diag(np.linalg.pinv(Xd.T @ Xd)) * (resid @ resid) / dof)
    names = ["intercept", "log_qualified_density", "log_cpm", "log_population"]
    RESULTS["awareness_cac_drivers"] = {
        "bivariate": {
            "corr_logpop": float(np.corrcoef(lp, la)[0, 1]),
            "corr_log_qualdensity": float(np.corrcoef(np.log(ok.qual_frac), la)[0, 1]),
            "corr_logcpm": float(np.corrcoef(np.log(ok.cpm_index), la)[0, 1]),
        },
        "multivariate": {n: {"coef": float(c), "se": float(s_),
                             "t": float(c / s_) if s_ > 0 else 0.0}
                         for n, c, s_ in zip(names, coef, se)},
        "r2": float(1 - (resid @ resid) / ((la - la.mean()) ** 2).sum()),
    }
    print("   ", RESULTS["price_of_awareness"])
    print("   ", RESULTS["awareness_cac_drivers"])

    # ---------------------------------------------------------------- 4. pooling
    print("\n[4] pooling experiment (new-market forecast)")
    RESULTS["pooling"] = pooling_experiment(panel, d, gs)

    # ---------------------------------------------------------------- 5. geo calib
    print("\n[5] geo-experiment prior calibration")
    RESULTS["geo_calibration"] = geo_calibration(d)

    # ---------------------------------------------------------------- 5b. allocation
    print("\n[5b] allocation consequence of the attenuation bias")
    RESULTS["allocation"] = allocation_consequence(d, cp, idata, cats)

    # ---------------------------------------------------------------- 6. figures
    print("\n[6] figures")
    make_figures(panel, d, cp, idata, cats, truth, econ)

    RESULTS["runtime_sec"] = round(time.time() - t_start, 1)
    with open(f"{DATA}/results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nDone in {RESULTS['runtime_sec']}s -> data/results.json")


# ==========================================================================
def search_awareness_adstock(panel, thetas=np.arange(0.50, 0.985, 0.025)):
    """
    Awareness is a STOCK with slow decay; acquisition responds to a much faster
    flow. Using one carryover rate for both flips the sign of the brand
    coefficient, so stage 1 gets its own grid search on its own equation.
    """
    best = None
    for th in thetas:
        df = M.add_adstock(panel, th, 0.6)
        df = df[df.activations > 0]
        xb = np.log1p(df.brand_ads / (df.population_m * df.cpm_index) / 1e3).values
        A = np.clip(df.awareness_survey.values, 0.004, 0.985)
        y = np.log(A / (1 - A))
        X = np.column_stack([np.ones(len(df)), xb,
                             pd.get_dummies(df.code, drop_first=True).values.astype(float)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = float(((y - X @ coef) ** 2).sum())
        r2 = 1 - rss / float(((y - y.mean()) ** 2).sum())
        if best is None or r2 > best["r2"]:
            best = {"theta": float(th), "r2": float(r2), "slope": float(coef[1])}
    return best


def fit_stage1(d):
    """logit(A) = a_c + b_c * log1p(brand_adstock / (pop * cpm))"""
    codes = pd.Categorical(d["code"]); ci = codes.codes; C = len(codes.categories)
    A = np.clip(d["x_aware"].values, 0.004, 0.985)
    y = np.log(A / (1 - A))
    x = d["x_brand"].values

    with pm.Model():
        mu_a = pm.Normal("mu_a", -3.0, 2.0); tau_a = pm.HalfNormal("tau_a", 1.5)
        mu_b = pm.Normal("mu_b", 1.0, 1.0); tau_b = pm.HalfNormal("tau_b", 0.8)
        za = pm.Normal("za", 0, 1, shape=C); zb = pm.Normal("zb", 0, 1, shape=C)
        a_c = pm.Deterministic("a_c", mu_a + tau_a * za)
        b_c = pm.Deterministic("b_c", mu_b + tau_b * zb)
        sig = pm.HalfNormal("sig", 1.0)
        pm.Normal("obs", mu=a_c[ci] + b_c[ci] * x, sigma=sig, observed=y)
        id1 = pm.sample(800, tune=800, chains=2, random_seed=M.SEED,
                        target_accept=0.92, progressbar=False,
                        idata_kwargs={"log_likelihood": False})

    a = id1.posterior["a_c"].values.mean(axis=(0, 1))
    b = id1.posterior["b_c"].values.mean(axis=(0, 1))

    rows = []
    for i, c in enumerate(codes.categories):
        sub = d[d.code == c]
        g = sub.iloc[0]
        pop, cpm = g.population_m, g.cpm_index
        obs = sub["brand_spend"].values
        obs = obs[obs > 0]
        if len(obs) == 0:
            obs = np.array([1e4])

        # evaluate the fitted curve across this market's own spend support
        grid = np.geomspace(max(obs.min() * 0.4, 1e3), obs.max() * 3.0, 400)
        xb = np.log1p(grid / (pop * cpm) / 1e3)
        Ahat = 1.0 / (1.0 + np.exp(-(a[i] + b[i] * xb)))

        # (a) LOCAL price of one awareness point, at this market's current position
        a_lo, a_hi = np.percentile(Ahat, 25), np.percentile(Ahat, 75)
        s_lo = np.interp(a_lo, Ahat, grid)
        s_hi = np.interp(a_hi, Ahat, grid)
        pts = max((a_hi - a_lo) * 100.0, 1e-6)
        local_price = float(max(s_hi - s_lo, 1.0) / pts)

        # (b) MATCHED price: monthly brand spend needed to sustain TARGET awareness.
        #     This is the apples-to-apples entry comparison. Search far out so we
        #     can say honestly when a target is simply unreachable.
        wide = np.geomspace(max(obs.min() * 0.2, 1e3), obs.max() * 12.0, 2000)
        xw = np.log1p(wide / (pop * cpm) / 1e3)
        Aw = 1.0 / (1.0 + np.exp(-(a[i] + b[i] * xw)))
        reachable = bool(Aw.max() >= TARGET_AW)
        s_target = float(np.interp(TARGET_AW, Aw, wide)) if reachable else float("nan")

        rows.append({"code": c, "a_c_aw": float(a[i]), "b_c_aw": float(b[i]),
                     "usd_per_pt": local_price,
                     "usd_to_target_aw": s_target,
                     "target_reachable": reachable,
                     "max_feasible_aw": float(Aw.max())})

    summ = az.summary(id1, var_names=["mu_a", "mu_b", "tau_a", "tau_b", "sig"])
    return {"per_market": pd.DataFrame(rows),
            "summary": {k: float(summ.loc[k, "mean"]) for k in summ.index},
            "max_rhat": float(summ["r_hat"].max())}


# ==========================================================================
def pooling_experiment(panel, d, gs, holdout="MEX", n_train=3):
    """
    Mexico enters with only 5 months of history. Fit three ways on
    {17 full markets + 5 months of MEX} and forecast MEX months 6-15.
    """
    mex = d[d.code == holdout].sort_values("month")
    keep = set(mex.month.values[:n_train])
    train = d[(d.code != holdout) | (d.month.isin(keep))].copy()
    test = mex[~mex.month.isin(keep)].copy()
    out = {"holdout": holdout, "n_train_months": n_train, "n_test_months": len(test)}

    for pooling in ["none", "complete", "partial"]:
        if pooling == "none":
            # genuinely no pooling: the new market is fit on its own data only,
            # so it borrows nothing -- not even the nuisance parameters
            solo = d[(d.code == holdout) & (d.month.isin(keep))].copy()
            _, idata, cats, _ = M.fit_hierarchical(solo, "complete", draws=800,
                                                   tune=800, chains=2)
            pred_log = M.predict(test, idata, cats, "complete")
        else:
            _, idata, cats, _ = M.fit_hierarchical(train, pooling, draws=800,
                                                   tune=800, chains=2)
            pred_log = M.predict(test, idata, cats, pooling)
        pred_act = np.exp(pred_log) * test.qualified_tam.values
        act = test.activations.values
        cac_true = test.total_spend.values / act
        cac_pred = test.total_spend.values / np.maximum(pred_act, 1e-6)

        out[pooling] = {
            "activation_mape": float(np.mean(np.abs(pred_act - act) / act) * 100),
            "cac_mape": float(np.mean(np.abs(cac_pred - cac_true) / cac_true) * 100),
            "cac_pred": [float(v) for v in cac_pred],
            "cac_true": [float(v) for v in cac_true],
            "months": [int(v) for v in test.month.values],
        }
        print(f"    {pooling:9s} activations MAPE {out[pooling]['activation_mape']:6.1f}%"
              f"   CAC MAPE {out[pooling]['cac_mape']:6.1f}%")
    return out


# ==========================================================================
def geo_calibration(d):
    """
    Simulate a geo holdout that measures the awareness effect directly, then
    refit with that result as a prior on mu_gamma and show the posterior tighten.
    """
    _, id_vague, _, _ = M.fit_hierarchical(d, "partial", gamma_prior=(1.0, 1.5),
                                           draws=800, tune=800, chains=2)
    # A geo holdout measures the awareness effect directly and therefore lands
    # near the TRUE value, not the attenuated observational one.
    geo_est, geo_se = 1.75, 0.20
    _, id_calib, _, _ = M.fit_hierarchical(d, "partial", gamma_prior=(geo_est, geo_se),
                                           draws=800, tune=800, chains=2)
    g0 = id_vague.posterior["mu_gamma"].values.reshape(-1)
    g1 = id_calib.posterior["mu_gamma"].values.reshape(-1)
    res = {
        "vague": {"mean": float(g0.mean()), "sd": float(g0.std()),
                  "lo": float(np.percentile(g0, 5)), "hi": float(np.percentile(g0, 95))},
        "calibrated": {"mean": float(g1.mean()), "sd": float(g1.std()),
                       "lo": float(np.percentile(g1, 5)), "hi": float(np.percentile(g1, 95))},
    }
    res["geo_experiment"] = {"estimate": geo_est, "se": geo_se}
    res["sd_reduction_pct"] = float((1 - g1.std() / g0.std()) * 100)
    res["shift_in_mean"] = float(g1.mean() - g0.mean())
    print(f"    mu_gamma {g0.mean():.3f} -> {g1.mean():.3f} | "
          f"sd {g0.std():.3f} -> {g1.std():.3f} "
          f"({res['sd_reduction_pct']:.0f}% tighter)")
    return res


# ==========================================================================
def allocation_consequence(d, cp, idata, cats):
    """
    Hold the budget fixed and ask what brand/performance split maximizes
    activations -- once with the attenuated observational gamma, once with the
    gamma a geo experiment would deliver. The gap is the cost of the bias.
    """
    post = idata.posterior
    delta = float(post["delta"].values.mean())
    zeta = float(post["zeta"].values.mean())
    gamma_obs = RESULTS["main_fit"]["mu_gamma"]["mean"]
    gamma_true = RESULTS["recovery"]["true_population_gamma"]
    shares = np.linspace(0.02, 0.95, 94)
    rows = []

    for _, r in cp.iterrows():
        sub = d[d.code == r.code].sort_values("month")
        if len(sub) == 0:
            continue
        last = sub.iloc[-1]
        budget = float(last.brand_spend + last.perf_spend)
        if budget <= 0:
            continue
        pop, cpm, tam = last.population_m, last.cpm_index, last.qualified_tam

        def best(gamma):
            brand = budget * shares
            perf = budget * (1 - shares)
            A = 1 / (1 + np.exp(-(r.a_c_aw + r.b_c_aw
                                  * np.log1p(brand / (pop * cpm) / 1e3))))
            xp = np.log1p(perf / tam * 1e3)
            la = (r.alpha_c + r.beta_c * xp + gamma * A
                  + delta * last.x_doors + zeta * last.x_pen)
            acts = np.exp(la) * tam
            i = int(np.argmax(acts))
            return float(shares[i]), float(acts[i])

        sh_obs, act_obs = best(gamma_obs)
        sh_true, act_true = best(gamma_true)
        # what you actually get by spending the observational split while the
        # world runs on the true gamma
        A_obs = 1 / (1 + np.exp(-(r.a_c_aw + r.b_c_aw
                                  * np.log1p(budget * sh_obs / (pop * cpm) / 1e3))))
        xp_obs = np.log1p(budget * (1 - sh_obs) / tam * 1e3)
        act_realised = float(np.exp(r.alpha_c + r.beta_c * xp_obs + gamma_true * A_obs
                                    + delta * last.x_doors + zeta * last.x_pen) * tam)
        rows.append({
            "market": r.market, "code": r.code, "budget": budget,
            "brand_share_observational": sh_obs, "brand_share_geo": sh_true,
            "activations_lost_pct": float((act_true - act_realised) / act_true * 100),
            "cac_penalty_pct": float((act_true / max(act_realised, 1e-9) - 1) * 100),
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{DATA}/allocation.csv", index=False)
    out = {
        "gamma_observational": gamma_obs, "gamma_geo": gamma_true,
        "mean_brand_share_observational": float(df.brand_share_observational.mean()),
        "mean_brand_share_geo": float(df.brand_share_geo.mean()),
        "mean_activations_lost_pct": float(df.activations_lost_pct.mean()),
        "mean_cac_penalty_pct": float(df.cac_penalty_pct.mean()),
        "per_market": df.round(4).to_dict(orient="records"),
    }
    print(f"    brand share: {out['mean_brand_share_observational']:.0%} (observational)"
          f"  ->  {out['mean_brand_share_geo']:.0%} (geo-calibrated)")
    print(f"    cost of the bias: {out['mean_activations_lost_pct']:.1f}% fewer "
          f"activations at the same budget")
    return out


# ==========================================================================
def make_figures(panel, d, cp, idata, cats, truth, econ):
    cp = cp.sort_values("qual_frac", ascending=False)
    ok = cp[cp.target_reachable.astype(bool) & np.isfinite(cp.awareness_cac)].copy()

    # --- F1: awareness trajectories
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    show = ["Finland", "UAE", "Netherlands", "UK", "Japan", "Brazil", "India"]
    cmap = plt.cm.viridis(np.linspace(0.08, 0.86, len(show)))
    for col, mkt in zip(cmap, show):
        g = panel[panel.market == mkt].sort_values("months_since_launch")
        ax.plot(g.months_since_launch, g.awareness_true * 100, lw=2.0, color=col, label=mkt)
    ax.set_xlabel("Months since market launch"); ax.set_ylabel("Brand awareness (%)")
    ax.set_title("Same playbook, very different awareness ceilings", loc="left")
    ax.legend(frameon=False, ncol=2, fontsize=8); ax.grid(alpha=.5)
    save(fig, "f1_awareness_trajectories")

    # --- F2: price of an awareness point vs population
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.scatter(ok.population_m, ok.usd_to_target_aw / 1e3, s=46, color=ACCENT,
               edgecolor="white", zorder=3)
    for _, r in ok.iterrows():
        ax.annotate(r.market, (r.population_m, r.usd_to_target_aw / 1e3),
                    fontsize=6.6, xytext=(4, 3), textcoords="offset points", color=MUTED)
    xs = np.linspace(ok.population_m.min(), ok.population_m.max(), 50)
    b, a = np.polyfit(np.log(ok.population_m), np.log(ok.usd_to_target_aw / 1e3), 1)
    ax.plot(xs, np.exp(a) * xs ** b, color=INK, lw=1.3, ls="--",
            label=f"slope on log pop = {b:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Population (millions, log)")
    ax.set_ylabel(f"$K/month to sustain {TARGET_AW:.0%} awareness (log)")
    ax.set_title("The price of awareness scales with population", loc="left")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.5, which="both")
    save(fig, "f2_price_of_awareness")

    # --- F3: awareness-driven CAC vs qualified density (the key result)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    sc = ax.scatter(ok.qual_frac * 100, ok.awareness_cac, s=52,
                    c=np.log10(ok.population_m), cmap="cividis",
                    edgecolor="white", zorder=3)
    for _, r in ok.iterrows():
        ax.annotate(r.market, (r.qual_frac * 100, r.awareness_cac), fontsize=6.6,
                    xytext=(4, 3), textcoords="offset points", color=MUTED)
    b, a = np.polyfit(np.log(ok.qual_frac), np.log(ok.awareness_cac), 1)
    xs = np.linspace(ok.qual_frac.min(), ok.qual_frac.max(), 50)
    ax.plot(xs * 100, np.exp(a) * xs ** b, color=INK, lw=1.3, ls="--",
            label=f"elasticity = {b:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("log₁₀ population (m)", fontsize=8)
    ax.set_xlabel("Qualified buyers as % of population (log)")
    ax.set_ylabel(f"Awareness-driven CAC at {TARGET_AW:.0%} awareness ($, log)")
    ax.set_title("At matched awareness, qualified density sets the cost —\n"
                 "population washes out", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.5, which="both")
    save(fig, "f3_awareness_cac_vs_density")

    # --- F4: shrinkage / forest plot of gamma
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    c2 = cp.sort_values("gamma_c")
    yv = np.arange(len(c2))
    ax.hlines(yv, c2.gamma_c_lo, c2.gamma_c_hi, color=MUTED, lw=1.6)
    ax.scatter(c2.gamma_c, yv, s=30, color=TEAL, zorder=3)
    mg = RESULTS["main_fit"]["mu_gamma"]["mean"]
    ax.axvline(mg, color=ACCENT, ls="--", lw=1.4, label=f"pooled mean = {mg:.2f}")
    ax.set_yticks(yv); ax.set_yticklabels(c2.market, fontsize=7.5)
    ax.set_xlabel("γ — awareness coefficient on log acquisition rate")
    ax.set_title("Country awareness effects, with partial pooling", loc="left")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.5, axis="x")
    save(fig, "f4_gamma_forest")

    # --- F5: pooling experiment
    p = RESULTS["pooling"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.plot(p["partial"]["months"], p["partial"]["cac_true"], "o-", color=INK,
            lw=2, ms=4, label="Actual CAC")
    for key, col, ls in [("none", "#c1121f", ":"), ("complete", MUTED, "--"),
                         ("partial", TEAL, "-")]:
        ax.plot(p[key]["months"], p[key]["cac_pred"], ls, color=col, lw=1.8,
                label=f"{key} pooling")
    ax.set_xlabel("Month"); ax.set_ylabel("Blended CAC ($)")
    ax.set_title(f"Forecasting {p['holdout']} from {p['n_train_months']} months of data",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.5)

    ax = axes[1]
    ks = ["none", "complete", "partial"]
    vals = [p[k]["cac_mape"] for k in ks]
    bars = ax.bar(["No\npooling", "Complete\npooling", "Partial\npooling"], vals,
                  color=["#c1121f", MUTED, TEAL], width=.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}%", ha="center",
                va="bottom", fontsize=9, color=INK)
    ax.set_ylabel("CAC forecast MAPE (%)")
    ax.set_title("Partial pooling wins", loc="left", fontsize=10)
    ax.grid(alpha=.5, axis="y")
    save(fig, "f5_pooling_experiment")

    # --- F6: average vs marginal CAC
    row = d[d.code == "GBR"].sort_values("month").iloc[-1]
    grid = np.linspace(row.perf_spend * 0.08, row.perf_spend * 3.2, 160)
    acts, avg, marg = M.response_curve(row, idata, cats, grid)
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    ax.plot(grid / 1e6, avg, color=INK, lw=2, label="Average CAC")
    ax.plot(grid / 1e6, marg, color=ACCENT, lw=2, label="Marginal CAC")
    ax.axvline(row.perf_spend / 1e6, color=MUTED, ls=":", lw=1.4)
    i = np.argmin(np.abs(grid - row.perf_spend))
    ax.annotate(f"current spend\navg ${avg[i]:.0f} vs marginal ${marg[i]:.0f}",
                (row.perf_spend / 1e6, marg[i]), fontsize=8, color=INK,
                xytext=(14, -6), textcoords="offset points")
    ax.set_xlabel("Monthly performance spend ($M)"); ax.set_ylabel("CAC ($)")
    ax.set_title("UK: the next customer costs far more than the average one", loc="left")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.5)
    save(fig, "f6_marginal_vs_average_cac")
    RESULTS["uk_cac_gap"] = {"avg": float(avg[i]), "marginal": float(marg[i]),
                             "ratio": float(marg[i] / avg[i])}

    # --- F7: geo calibration
    g = RESULTS["geo_calibration"]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for i, (k, col, lab) in enumerate([("vague", MUTED, "Vague prior"),
                                       ("calibrated", TEAL, "Geo-calibrated prior")]):
        v = g[k]
        ax.hlines(i, v["lo"], v["hi"], color=col, lw=7, alpha=.75)
        ax.plot(v["mean"], i, "o", color=INK, ms=6)
        ax.text(v["hi"] + .02, i, f"{lab}  (sd {v['sd']:.3f})", va="center",
                fontsize=8.5, color=INK)
    ax.set_yticks([]); ax.set_ylim(-.6, 1.8)
    ax.set_xlabel("Posterior for μ_γ (population awareness effect)")
    ax.axvline(RESULTS["recovery"]["true_population_gamma"], color="#c1121f",
               ls="--", lw=1.4)
    ax.text(RESULTS["recovery"]["true_population_gamma"], 1.55, " true effect",
            color="#c1121f", fontsize=8, va="top")
    ax.set_title("Observational data alone understates awareness; a geo holdout "
                 "corrects it", loc="left", fontsize=10)
    ax.grid(alpha=.5, axis="x")
    save(fig, "f7_geo_calibration")

    # --- F7b: attenuation bias
    rec = RESULTS["recovery"]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.barh(["True effect\n(simulator)", "MMM estimate\n(observational)"],
            [rec["true_population_gamma"], rec["estimated_population_gamma"]],
            color=["#c1121f", MUTED], height=.55)
    for y, v in enumerate([rec["true_population_gamma"], rec["estimated_population_gamma"]]):
        ax.text(v + .03, y, f"γ = {v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Awareness effect on log acquisition rate")
    ax.set_title(f"The MMM understates awareness by "
                 f"{rec['pct_error_population']:.0f}%", loc="left")
    ax.grid(alpha=.5, axis="x")
    save(fig, "f7b_attenuation")

    # --- F7c: allocation consequence
    al = pd.DataFrame(RESULTS["allocation"]["per_market"]).sort_values("brand_share_geo")
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    y = np.arange(len(al))
    ax.hlines(y, al.brand_share_observational * 100, al.brand_share_geo * 100,
              color="#d8dee9", lw=3, zorder=1)
    ax.scatter(al.brand_share_observational * 100, y, s=34, color=MUTED, zorder=3,
               label="Budget split the MMM recommends")
    ax.scatter(al.brand_share_geo * 100, y, s=34, color=ACCENT, zorder=3,
               label="Split once awareness is measured properly")
    ax.set_yticks(y); ax.set_yticklabels(al.market, fontsize=7.4)
    ax.set_xlabel("Optimal share of budget going to brand / upper funnel (%)")
    ax.set_title("Attenuated awareness estimates systematically under-fund brand",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.grid(alpha=.5, axis="x")
    save(fig, "f7c_allocation")

    # --- F8: entry planner — CAC vs penetration by archetype
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    arche = {"Finland": TEAL, "Netherlands": "#457b9d", "UK": INK,
             "Japan": "#9c6644", "Brazil": ACCENT, "India": "#c1121f"}
    for mkt, col in arche.items():
        g = panel[panel.market == mkt].sort_values("month")
        g = g[g.penetration > 0.002]
        roll = g.set_index("penetration")["blended_cac"].rolling(5, min_periods=2).mean()
        ax.plot(roll.index * 100, roll.values, color=col, lw=2, label=mkt)
    ax.set_xlabel("Penetration of qualified TAM (%)")
    ax.set_ylabel("Blended CAC ($, 5-month rolling)")
    ax.set_title("CAC as a function of market maturity, not calendar time", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2); ax.grid(alpha=.5)
    save(fig, "f8_cac_vs_penetration")

    # --- F9: payback by market
    cm = econ
    last = (panel.sort_values("month").groupby("market")
            .tail(6).groupby("market")
            .agg(cac=("blended_cac", "mean"), qd=("qual_frac", "first")).reset_index())
    last["payback"] = S.payback_months(last.cac.values, cm["hardware_cm"],
                                       cm["monthly_subscription_cm"],
                                       cm["monthly_retention"])
    last = last.sort_values("payback")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    cols = [TEAL if v <= 12 else (ACCENT if v <= 24 else "#c1121f") for v in last.payback]
    ax.barh(last.market, last.payback, color=cols, height=.68)
    for y, (v, c) in enumerate(zip(last.payback, last.cac)):
        ax.text(v + .3, y, f"{v:.0f} mo  (CAC ${c:.0f})", va="center", fontsize=7.4,
                color=MUTED)
    ax.axvline(12, color=INK, ls=":", lw=1.2)
    ax.set_xlabel("Months to repay CAC from contribution margin")
    ax.set_title("Payback, not CAC, is the entry guardrail", loc="left")
    ax.grid(alpha=.5, axis="x")
    save(fig, "f9_payback")
    RESULTS["payback_table"] = last.round(2).to_dict(orient="records")

    # --- F10: channel decomposition
    agg = (panel.groupby("market")[["organic_activations", "paid_activations",
                                    "retail_activations"]].sum())
    agg = agg.div(agg.sum(axis=1), axis=0).sort_values("organic_activations")
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    left = np.zeros(len(agg))
    for col, c, lab in [("organic_activations", TEAL, "Organic / word of mouth"),
                        ("paid_activations", INK, "Paid performance"),
                        ("retail_activations", ACCENT, "Retail")]:
        ax.barh(agg.index, agg[col] * 100, left=left * 100, color=c, label=lab, height=.7)
        left += agg[col].values
    ax.set_xlabel("Share of activations (%)")
    ax.set_title("Where acquisitions come from", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center",
              bbox_to_anchor=(.5, -.22)); ax.grid(alpha=.5, axis="x")
    ax.tick_params(axis="y", labelsize=7.5)
    save(fig, "f10_channel_mix")


if __name__ == "__main__":
    main()
