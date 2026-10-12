"""
Hierarchical Bayesian marketing mix model with awareness mediation.

The model only ever sees quantities a real team would have:
  spend by channel, a noisy awareness proxy, activations, country covariates.
It never sees the latent awareness or the true parameters in simulate.py.

Structure
---------
Stage 1 (awareness response):
    logit(A_ct) = a_c + b_c * log1p(brand_adstock_ct / (pop_c * cpm_c))

Stage 2 (acquisition response):
    log(activations_ct / TAM_c) = alpha_c
                                + beta_c  * log1p(perf_adstock_ct / TAM_c)
                                + gamma_c * A_ct
                                + delta   * log1p(doors_ct / TAM_c)
                                + zeta    * penetration_ct
                                + seasonality_t
                                + eps

Partial pooling:
    alpha_c ~ N(mu_alpha, tau_alpha)
    beta_c  ~ N(mu_beta,  tau_beta)
    gamma_c ~ N(mu_gamma + kappa * z_qualdensity_c, tau_gamma)

kappa is the headline parameter: it says whether the awareness elasticity of
acquisition depends on how dense a market's qualified buyers are.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

SEED = 20261012


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def adstock_norm(x, theta):
    out = np.zeros(len(x))
    acc = 0.0
    for t in range(len(x)):
        acc = x[t] + theta * acc
        out[t] = acc
    return out * (1 - theta)


def add_adstock(panel, theta_brand, theta_perf):
    df = panel.sort_values(["code", "month"]).copy()
    df["brand_ads"] = (df.groupby("code")["brand_spend"]
                         .transform(lambda s: adstock_norm(s.values, theta_brand)))
    df["perf_ads"] = (df.groupby("code")["perf_spend"]
                        .transform(lambda s: adstock_norm(s.values, theta_perf)))
    return df


def build_features(panel, theta_brand=0.60, theta_perf=0.55, awareness_col="awareness_survey"):
    df = add_adstock(panel, theta_brand, theta_perf)
    df = df[df["activations"] > 0].copy()

    df["x_perf"] = np.log1p(df["perf_ads"] / df["qualified_tam"] * 1e3)
    df["x_brand"] = np.log1p(df["brand_ads"] / (df["population_m"] * df["cpm_index"]) / 1e3)
    df["x_aware"] = df[awareness_col]
    df["doors_live"] = df["doors"] * np.minimum(1.0, (df["months_since_launch"] + 1) / 12.0)
    df["x_doors"] = np.log1p(df["doors_live"] / df["qualified_tam"] * 1e6)
    df["x_pen"] = df["penetration"]
    df["y"] = np.log(df["activations"] / df["qualified_tam"])

    moy = df["month"] % 12
    df["sin1"] = np.sin(2 * np.pi * moy / 12)
    df["cos1"] = np.cos(2 * np.pi * moy / 12)
    return df


def grid_search_adstock(panel, thetas=np.arange(0.30, 0.86, 0.05)):
    """
    Pick carryover rates outside the Bayesian core by pooled-OLS BIC.
    This is the standard pragmatic approach (Robyn does an equivalent search);
    it keeps the sampler fast and the hyperparameters identifiable.
    """
    best = None
    for tb in thetas:
        for tp in thetas:
            d = build_features(panel, tb, tp)
            X = np.column_stack([
                np.ones(len(d)), d.x_perf, d.x_aware, d.x_doors,
                d.x_pen, d.sin1, d.cos1,
                pd.get_dummies(d.code, drop_first=True).values.astype(float)
            ])
            y = d.y.values
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            rss = float(((y - X @ coef) ** 2).sum())
            n, k = len(y), X.shape[1]
            bic = n * np.log(rss / n) + k * np.log(n)
            if best is None or bic < best[0]:
                best = (bic, float(tb), float(tp))
    return {"bic": best[0], "theta_brand": best[1], "theta_perf": best[2]}


# --------------------------------------------------------------------------
# The hierarchical model
# --------------------------------------------------------------------------
def fit_hierarchical(d, pooling="partial", gamma_prior=None,
                     draws=1000, tune=1000, chains=2, seed=SEED, progress=False):
    """
    pooling : "partial" (hierarchical), "none" (independent per country),
              "complete" (one global coefficient set)
    gamma_prior : optional (mu, sd) to encode a geo-experiment result on the
              population-level awareness effect.
    """
    codes = pd.Categorical(d["code"])
    ci = codes.codes
    C = len(codes.categories)

    # country-level covariate: standardized log qualified density
    qd = (d.groupby("code")["qual_frac"].first()
            .reindex(codes.categories).values)
    _sd = np.log(qd).std()
    z_qd = ((np.log(qd) - np.log(qd).mean()) / _sd) if _sd > 0 else np.zeros_like(qd)

    X_perf = d["x_perf"].values
    X_aware = d["x_aware"].values
    X_doors = d["x_doors"].values
    X_pen = d["x_pen"].values
    S1, S2 = d["sin1"].values, d["cos1"].values
    y = d["y"].values

    gmu, gsd = (gamma_prior if gamma_prior is not None else (1.0, 1.5))

    with pm.Model() as model:
        if pooling == "complete":
            alpha_c = pm.Normal("alpha_c", -6.0, 2.0)
            beta_c = pm.Normal("beta_c", 0.5, 1.0)
            gamma_c = pm.Normal("gamma_c", gmu, gsd)
            a_idx = alpha_c
            b_idx = beta_c
            g_idx = gamma_c
            pm.Deterministic("mu_gamma", gamma_c)

        elif pooling == "none":
            alpha_c = pm.Normal("alpha_c", -6.0, 3.0, shape=C)
            beta_c = pm.Normal("beta_c", 0.5, 2.0, shape=C)
            gamma_c = pm.Normal("gamma_c", gmu, 3.0, shape=C)
            a_idx, b_idx, g_idx = alpha_c[ci], beta_c[ci], gamma_c[ci]
            pm.Deterministic("mu_gamma", pm.math.mean(gamma_c))

        else:  # partial pooling
            mu_alpha = pm.Normal("mu_alpha", -6.0, 2.0)
            tau_alpha = pm.HalfNormal("tau_alpha", 1.0)
            mu_beta = pm.Normal("mu_beta", 0.5, 1.0)
            tau_beta = pm.HalfNormal("tau_beta", 0.5)
            mu_gamma = pm.Normal("mu_gamma", gmu, gsd)
            tau_gamma = pm.HalfNormal("tau_gamma", 0.7)
            kappa = pm.Normal("kappa", 0.0, 1.0)   # qualified-density moderator

            za = pm.Normal("za", 0, 1, shape=C)
            zb = pm.Normal("zb", 0, 1, shape=C)
            zg = pm.Normal("zg", 0, 1, shape=C)

            alpha_c = pm.Deterministic("alpha_c", mu_alpha + tau_alpha * za)
            beta_c = pm.Deterministic("beta_c", mu_beta + tau_beta * zb)
            gamma_c = pm.Deterministic("gamma_c",
                                       mu_gamma + kappa * z_qd + tau_gamma * zg)
            a_idx, b_idx, g_idx = alpha_c[ci], beta_c[ci], gamma_c[ci]

        delta = pm.Normal("delta", 0.0, 1.0)
        zeta = pm.Normal("zeta", 0.0, 2.0)
        s_sin = pm.Normal("s_sin", 0.0, 0.5)
        s_cos = pm.Normal("s_cos", 0.0, 0.5)
        sigma = pm.HalfNormal("sigma", 0.5)

        mu = (a_idx + b_idx * X_perf + g_idx * X_aware
              + delta * X_doors + zeta * X_pen + s_sin * S1 + s_cos * S2)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                          target_accept=0.92, progressbar=progress,
                          idata_kwargs={"log_likelihood": False})

    return model, idata, list(codes.categories), z_qd


def posterior_country_params(idata, categories, pooling="partial"):
    """Tidy table of per-country posterior means and 90% intervals."""
    post = idata.posterior
    out = []
    for i, c in enumerate(categories):
        row = {"code": c}
        for name in ["alpha_c", "beta_c", "gamma_c"]:
            v = post[name]
            arr = v.values.reshape(-1) if v.values.ndim == 2 else v.values[:, :, i].reshape(-1)
            row[name] = float(arr.mean())
            row[name + "_lo"] = float(np.percentile(arr, 5))
            row[name + "_hi"] = float(np.percentile(arr, 95))
        out.append(row)
    return pd.DataFrame(out)


def predict(d, idata, categories, pooling="partial"):
    """Posterior-mean prediction of log acquisition rate."""
    post = idata.posterior
    idx = {c: i for i, c in enumerate(categories)}

    def getp(name, code):
        v = post[name].values
        if v.ndim == 2:          # complete pooling, scalar
            return float(v.mean())
        return float(v[:, :, idx[code]].mean())

    a = np.array([getp("alpha_c", c) for c in d["code"]])
    b = np.array([getp("beta_c", c) for c in d["code"]])
    g = np.array([getp("gamma_c", c) for c in d["code"]])
    delta = float(post["delta"].values.mean())
    zeta = float(post["zeta"].values.mean())
    ssin = float(post["s_sin"].values.mean())
    scos = float(post["s_cos"].values.mean())

    return (a + b * d["x_perf"].values + g * d["x_aware"].values
            + delta * d["x_doors"].values + zeta * d["x_pen"].values
            + ssin * d["sin1"].values + scos * d["cos1"].values)


def predict_new_market(d_new, idata, categories, z_qd_new, pooling="partial"):
    """
    Prediction for a market the model has NOT seen. Partial pooling falls back
    to the population mean plus the qualified-density moderator; complete
    pooling uses the single global coefficient set; no-pooling has nothing to
    fall back to and uses the unweighted average of fitted countries.
    """
    post = idata.posterior
    if pooling == "partial":
        a = float(post["mu_alpha"].values.mean())
        b = float(post["mu_beta"].values.mean())
        g = float(post["mu_gamma"].values.mean()) + float(post["kappa"].values.mean()) * z_qd_new
    elif pooling == "complete":
        a = float(post["alpha_c"].values.mean())
        b = float(post["beta_c"].values.mean())
        g = float(post["gamma_c"].values.mean())
    else:
        a = float(post["alpha_c"].values.mean())
        b = float(post["beta_c"].values.mean())
        g = float(post["gamma_c"].values.mean())

    delta = float(post["delta"].values.mean())
    zeta = float(post["zeta"].values.mean())
    ssin = float(post["s_sin"].values.mean())
    scos = float(post["s_cos"].values.mean())
    return (a + b * d_new["x_perf"].values + g * d_new["x_aware"].values
            + delta * d_new["x_doors"].values + zeta * d_new["x_pen"].values
            + ssin * d_new["sin1"].values + scos * d_new["cos1"].values)


# --------------------------------------------------------------------------
# Decision layer: marginal CAC
# --------------------------------------------------------------------------
def response_curve(row, idata, categories, spend_grid, pooling="partial"):
    """
    Activations as a function of performance spend, holding awareness and
    penetration fixed at the row's values. Returns (activations, avg CAC,
    marginal CAC) on the grid.
    """
    post = idata.posterior
    idx = {c: i for i, c in enumerate(categories)}
    v = post["beta_c"].values
    b = float(v.mean()) if v.ndim == 2 else float(v[:, :, idx[row["code"]]].mean())
    v = post["alpha_c"].values
    a = float(v.mean()) if v.ndim == 2 else float(v[:, :, idx[row["code"]]].mean())
    v = post["gamma_c"].values
    g = float(v.mean()) if v.ndim == 2 else float(v[:, :, idx[row["code"]]].mean())
    delta = float(post["delta"].values.mean())
    zeta = float(post["zeta"].values.mean())

    tam = row["qualified_tam"]
    x_perf = np.log1p(spend_grid / tam * 1e3)
    log_rate = (a + b * x_perf + g * row["x_aware"]
                + delta * row["x_doors"] + zeta * row["x_pen"])
    acts = np.exp(log_rate) * tam

    fixed = row["brand_spend"] + row.get("retail_spend", 0.0)
    total = spend_grid + fixed
    avg_cac = total / np.maximum(acts, 1e-9)
    d_act = np.gradient(acts, spend_grid)
    marg_cac = 1.0 / np.maximum(d_act, 1e-12)
    return acts, avg_cac, marg_cac
