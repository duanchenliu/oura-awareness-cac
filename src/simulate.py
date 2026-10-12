"""
Synthetic market generator for the awareness -> CAC study.

ALL DATA PRODUCED HERE IS SIMULATED. No Oura internal data was used.
Structural parameters (organic share, retention, international mix) are
calibrated to figures Oura disclosed publicly in its Form S-1 (filed
2026-09-03) so the synthetic world has a realistic shape; everything else
is invented and documented in PARAMS below.

Generative process
------------------
1. Reach      : saturating function of brand spend against local CPM x population
2. Awareness  : stock with decay, can only convert the not-yet-aware
3. Adoption   : Bass diffusion on a QUALIFIED TAM, where the innovation
                coefficient p is lifted by awareness and the imitation
                coefficient q drives word of mouth
4. Paid       : Hill-saturated response to performance spend, multiplied by an
                awareness-driven conversion lift, damped by remaining market
5. Retail     : doors x velocity, also lifted by awareness (the channel where
                awareness pays off invisibly to DTC attribution)
"""

import numpy as np
import pandas as pd

RNG_SEED = 20261012

# --------------------------------------------------------------------------
# Structural anchors taken from Oura's public Form S-1 (2026-09-03)
# --------------------------------------------------------------------------
SEC_ANCHORS = {
    "organic_share_of_new_subscribers": 0.40,
    "paid_member_12m_retention": 0.85,
    "intl_share_of_hardware_sales": 0.20,  # "less than 20%"
    "paying_subscribers_m": 5.0,
    "retail_doors": 8400,
    "international_markets": 56,
}

# Ring economics. INVENTED but set to publicly plausible street pricing.
RING_PRICE_USD = 349.0
HARDWARE_CM_RATE = 0.52          # contribution margin on hardware
MEMBERSHIP_MONTHLY_USD = 5.99
MEMBERSHIP_CM_RATE = 0.88        # near-zero marginal cost per subscriber
MEMBERSHIP_ATTACH = 0.80         # share of activations taking membership
MONTHLY_RETENTION = SEC_ANCHORS["paid_member_12m_retention"] ** (1 / 12)

N_MONTHS = 36


# --------------------------------------------------------------------------
# Market definitions
#
# population        : millions
# gdp_pc            : GDP per capita, thousands USD (PPP-ish)
# qual_frac         : share of population that is a QUALIFIED buyer -- can
#                     afford a $349 ring + subscription AND is in the
#                     health-tracking category. This is the parameter the whole
#                     study turns on.
# cpm_index         : local media cost index, US = 1.00
# doors             : retail points of sale
# comp              : competitive intensity 0-1 (Samsung, Ultrahuman, Whoop)
# decay             : monthly awareness forgetting rate
# q_bass            : word-of-mouth imitation coefficient
# launch_month      : month index the market opens (0 = already live)
# --------------------------------------------------------------------------
MARKETS = [
    # code  name           pop     gdp_pc qual   cpm   doors comp  decay  q     launch
    ("GBR", "UK",          68.3,   54.6,  0.085, 0.82,  620, 0.55, 0.045, 0.38,  0),
    ("DEU", "Germany",     83.3,   63.2,  0.072, 0.74,  540, 0.52, 0.050, 0.33,  0),
    ("CAN", "Canada",      39.1,   56.4,  0.090, 0.78,  380, 0.50, 0.042, 0.40,  0),
    ("AUS", "Australia",   26.7,   62.6,  0.095, 0.86,  290, 0.48, 0.040, 0.42,  0),
    ("FIN", "Finland",      5.6,   58.9,  0.135, 0.69,   70, 0.30, 0.030, 0.55,  0),
    ("SWE", "Sweden",      10.6,   64.1,  0.118, 0.72,  110, 0.35, 0.033, 0.50,  0),
    ("NLD", "Netherlands", 17.9,   72.4,  0.112, 0.76,  150, 0.40, 0.035, 0.46,  0),
    ("FRA", "France",      68.4,   55.5,  0.065, 0.77,  480, 0.50, 0.052, 0.31,  3),
    ("ESP", "Spain",       48.4,   46.4,  0.048, 0.61,  300, 0.45, 0.055, 0.34,  6),
    ("ITA", "Italy",       58.9,   49.3,  0.045, 0.63,  330, 0.44, 0.056, 0.32,  6),
    ("JPN", "Japan",      123.3,   45.6,  0.055, 0.95,  700, 0.68, 0.058, 0.26,  9),
    ("KOR", "South Korea", 51.7,   50.1,  0.068, 0.88,  420, 0.75, 0.054, 0.30,  9),
    ("ARE", "UAE",          9.5,   77.2,  0.125, 0.70,   95, 0.38, 0.036, 0.48, 12),
    ("SGP", "Singapore",    5.9,   91.1,  0.148, 0.83,   60, 0.42, 0.032, 0.52, 12),
    ("POL", "Poland",      36.7,   41.7,  0.034, 0.44,  210, 0.36, 0.060, 0.35, 15),
    ("BRA", "Brazil",     216.4,   18.1,  0.011, 0.38,  450, 0.40, 0.065, 0.44, 18),
    ("MEX", "Mexico",     128.5,   21.5,  0.013, 0.40,  380, 0.38, 0.063, 0.43, 21),
    ("IND", "India",     1428.6,    9.2,  0.0022, 0.21, 520, 0.58, 0.070, 0.47, 24),
]

MARKET_COLS = ["code", "market", "population_m", "gdp_pc_k", "qual_frac",
               "cpm_index", "doors", "competition", "awareness_decay",
               "q_bass", "launch_month"]

# --------------------------------------------------------------------------
# Global (shared) structural parameters -- these are the GROUND TRUTH the
# hierarchical model will try to recover.
# --------------------------------------------------------------------------
PARAMS = {
    # Reach: brand dollars needed per 1% reach per million population, at CPM index 1.0
    "reach_cost_per_pct_per_m": 1_850.0,
    # Awareness: fraction of reached-but-unaware who become aware
    "lambda_awareness": 0.62,
    # Bass innovation coefficient floor, MONTHLY (pure distribution/discovery)
    "p_base": 0.00012,
    # Awareness lift on the innovation coefficient -- the main causal channel
    "psi_awareness_to_p": 0.00165,
    # Performance media: Hill curve
    "hill_K_per_qual_m": 230_000.0,   # half-saturation spend scaled by qualified TAM (m)
    "hill_n": 1.05,
    "beta_perf": 0.00290,             # ceiling of paid acquisitions as share of qualified TAM
    # Awareness lift on paid conversion rate -- the second causal channel
    "eta_awareness_to_cvr": 1.20,
    # Retail: monthly units per door at full awareness, scaled by qualified density
    "retail_velocity": 7.0,
    "retail_trade_per_unit": 32.0,
    "retail_cost_per_door_mo": 260.0,
    "eta_awareness_to_retail": 1.60,
    # Competitive drag on all acquisition
    "comp_drag": 0.45,
    # Noise
    "obs_noise_sd": 0.065,
    "survey_noise_sd": 0.022,
    "sos_noise_sd": 0.055,
}


def hill(x, K, n):
    """Hill saturation. Returns value in [0, 1)."""
    x = np.maximum(x, 0.0)
    return x ** n / (K ** n + x ** n)


def adstock(x, theta, normalize=True):
    """Geometric carryover. Normalized so the output stays on the spend scale."""
    out = np.zeros_like(x, dtype=float)
    acc = 0.0
    for t in range(len(x)):
        acc = x[t] + theta * acc
        out[t] = acc
    return out * (1 - theta) if normalize else out


def build_spend_plan(rng, n_months, launch_month, qualified_tam, maturity_tier):
    """
    Media plan. Markets get a launch burst weighted to brand, then settle into
    a steadier mix. Budget scales with qualified TAM and local media cost, which
    is how a real planning team would size it.
    """
    brand = np.zeros(n_months)
    perf = np.zeros(n_months)

    # Monthly steady-state budget: anchored to qualified TAM and local CPM
    base = 0.25 * qualified_tam
    base *= {0: 1.00, 1: 0.78, 2: 0.60}[maturity_tier]

    for t in range(n_months):
        if t < launch_month:
            continue
        age = t - launch_month
        # Launch burst: 3x budget in month 0 decaying over ~6 months
        burst = 1.0 + 2.0 * np.exp(-age / 3.0)
        # Brand share starts high at launch (you must create awareness first)
        brand_share = 0.62 * np.exp(-age / 9.0) + 0.22
        total = base * burst * rng.lognormal(0.0, 0.14)
        # Q4 seasonality
        season = 1.0 + 0.35 * (np.cos(2 * np.pi * ((t % 12) - 11) / 12) > 0.80)
        total *= season
        brand[t] = total * brand_share
        perf[t] = total * (1 - brand_share)
    return brand, perf


def simulate(seed=RNG_SEED, n_months=N_MONTHS):
    rng = np.random.default_rng(seed)
    mk = pd.DataFrame(MARKETS, columns=MARKET_COLS)
    mk["qualified_tam"] = mk["population_m"] * 1e6 * mk["qual_frac"]

    # maturity tier drives budget intensity: 0 = flagship, 1 = core, 2 = emerging
    mk["maturity_tier"] = np.where(mk["launch_month"] == 0, 0,
                          np.where(mk["launch_month"] <= 12, 1, 2))

    rows = []
    truth = []

    for _, m in mk.iterrows():
        n = n_months
        qt = m["qualified_tam"]
        theta = 1.0 - m["awareness_decay"] * 3.0   # adstock retention for perf media
        theta = float(np.clip(theta, 0.45, 0.85))

        brand, perf = build_spend_plan(
            rng, n, int(m["launch_month"]), qt, int(m["maturity_tier"])
        )

        # --- country-specific deviations from the global params (what pooling learns)
        lam_c = PARAMS["lambda_awareness"] * rng.lognormal(0, 0.12)
        psi_c = PARAMS["psi_awareness_to_p"] * rng.lognormal(0, 0.18)
        eta_c = PARAMS["eta_awareness_to_cvr"] * rng.lognormal(0, 0.16)
        beta_c = PARAMS["beta_perf"] * rng.lognormal(0, 0.15)

        # Cost of one reach point rises with population (more people to reach)
        cost_per_pct_reach = (PARAMS["reach_cost_per_pct_per_m"]
                              * m["population_m"] * m["cpm_index"])

        awareness = np.zeros(n)
        cum_adopt = np.zeros(n)
        organic = np.zeros(n)
        paid = np.zeros(n)
        retail = np.zeros(n)
        reach_pct = np.zeros(n)

        perf_ads = adstock(perf, theta)
        hill_K = PARAMS["hill_K_per_qual_m"] * (qt / 1e6) * m["cpm_index"]
        comp_mult = 1.0 - PARAMS["comp_drag"] * m["competition"]

        A_prev, N_prev = 0.0, 0.0
        for t in range(n):
            if t < m["launch_month"]:
                continue

            # 1. reach from brand spend (saturating)
            reach_pct[t] = 1.0 - np.exp(-brand[t] / max(cost_per_pct_reach * 100.0, 1.0))

            # 2. awareness stock: decay + conversion of the not-yet-aware
            A = ((1 - m["awareness_decay"]) * A_prev
                 + lam_c * reach_pct[t] * (1 - A_prev))
            A = float(np.clip(A, 0, 0.97))
            awareness[t] = A

            remaining = max(qt - N_prev, 0.0)
            penetration = N_prev / qt if qt > 0 else 0.0

            # 3. organic: Bass diffusion, innovation coefficient lifted by awareness
            p_eff = PARAMS["p_base"] + psi_c * A
            q_eff = m["q_bass"] / 16.0   # annual -> monthly
            organic[t] = (p_eff + q_eff * penetration) * remaining * comp_mult

            # 4. paid: Hill response x awareness-driven CVR lift x remaining market
            cvr_lift = 1.0 + eta_c * A
            paid[t] = (beta_c * qt * hill(perf_ads[t], hill_K, PARAMS["hill_n"])
                       * cvr_lift * (1 - penetration) * comp_mult)

            # 5. retail: doors x velocity, also awareness-lifted
            door_ramp = min(1.0, (t - m["launch_month"] + 1) / 12.0)
            retail[t] = (m["doors"] * door_ramp * PARAMS["retail_velocity"]
                         * (0.12 + PARAMS["eta_awareness_to_retail"] * A)
                         * (1 - penetration) * comp_mult * (qt / (qt + 3e5)))

            total_t = organic[t] + paid[t] + retail[t]
            total_t *= rng.lognormal(0, PARAMS["obs_noise_sd"])
            total_t = min(total_t, remaining)

            # rescale components to the noised total
            raw = organic[t] + paid[t] + retail[t]
            if raw > 0:
                scale = total_t / raw
                organic[t] *= scale
                paid[t] *= scale
                retail[t] *= scale

            N_prev = N_prev + total_t
            cum_adopt[t] = N_prev
            A_prev = A

        activations = organic + paid + retail
        door_ramp_v = np.array([min(1.0, max(t - m["launch_month"] + 1, 0) / 12.0)
                                for t in range(n)])
        retail_spend = (retail * PARAMS["retail_trade_per_unit"]
                        + m["doors"] * door_ramp_v * PARAMS["retail_cost_per_door_mo"])
        retail_spend[brand + perf <= 0] = 0.0
        total_spend = brand + perf + retail_spend

        # Observed awareness signals
        survey = np.clip(awareness + rng.normal(0, PARAMS["survey_noise_sd"], n), 0, 1)
        share_of_search = np.clip(
            awareness * 0.72 * rng.lognormal(0, PARAMS["sos_noise_sd"], n), 0, 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            blended_cac = np.where(activations > 0, total_spend / activations, np.nan)
            paid_cac = np.where(paid > 0, perf / paid, np.nan)
            dtc_cac = np.where((activations - retail) > 0,
                               total_spend / (activations - retail), np.nan)
            organic_share = np.where(activations > 0,
                                     organic / activations, np.nan)
            retail_share = np.where(activations > 0,
                                    retail / activations, np.nan)

        for t in range(n):
            if t < m["launch_month"]:
                continue
            rows.append({
                "code": m["code"], "market": m["market"], "month": t,
                "months_since_launch": t - int(m["launch_month"]),
                "population_m": m["population_m"], "gdp_pc_k": m["gdp_pc_k"],
                "qual_frac": m["qual_frac"], "qualified_tam": qt,
                "cpm_index": m["cpm_index"], "doors": m["doors"],
                "competition": m["competition"], "maturity_tier": m["maturity_tier"],
                "brand_spend": brand[t], "perf_spend": perf[t],
                "retail_spend": retail_spend[t],
                "total_spend": total_spend[t],
                "reach_pct": reach_pct[t],
                "awareness_true": awareness[t],
                "awareness_survey": survey[t],
                "share_of_search": share_of_search[t],
                "activations": activations[t],
                "organic_activations": organic[t],
                "paid_activations": paid[t],
                "retail_activations": retail[t],
                "cum_activations": cum_adopt[t],
                "penetration": cum_adopt[t] / qt,
                "blended_cac": blended_cac[t],
                "paid_cac": paid_cac[t],
                "dtc_cac": dtc_cac[t],
                "organic_share": organic_share[t],
                "retail_share": retail_share[t],
            })

        truth.append({
            "code": m["code"], "market": m["market"],
            "lambda_c": lam_c, "psi_c": psi_c, "eta_c": eta_c, "beta_c": beta_c,
            "theta_adstock": theta, "hill_K": hill_K,
            "qualified_tam": qt, "qual_frac": m["qual_frac"],
            "cost_per_pct_reach": cost_per_pct_reach,
        })

    panel = pd.DataFrame(rows)
    truth = pd.DataFrame(truth)
    return panel, truth, mk


def unit_economics():
    """Contribution margin and payback inputs. All INVENTED but plausible."""
    hw_cm = RING_PRICE_USD * HARDWARE_CM_RATE
    sub_cm = MEMBERSHIP_MONTHLY_USD * MEMBERSHIP_CM_RATE * MEMBERSHIP_ATTACH
    return {
        "hardware_cm": hw_cm,
        "monthly_subscription_cm": sub_cm,
        "monthly_retention": MONTHLY_RETENTION,
    }


def payback_months(cac, hw_cm, monthly_cm, r, cap=60):
    """Months until cumulative contribution margin repays CAC."""
    cac = np.asarray(cac, dtype=float)
    out = np.full(cac.shape, np.nan)
    remaining = cac - hw_cm
    out[remaining <= 0] = 0.0
    idx = remaining > 0
    if idx.any():
        cum = 0.0
        for t in range(1, cap + 1):
            cum += monthly_cm * (r ** (t - 1))
            hit = idx & np.isnan(out) & (cum >= remaining)
            out[hit] = t
    return out


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    panel, truth, mk = simulate()
    panel.to_csv(f"{here}/data/panel.csv", index=False)
    truth.to_csv(f"{here}/data/ground_truth.csv", index=False)
    mk.to_csv(f"{here}/data/markets.csv", index=False)
    print(f"panel: {panel.shape}, markets: {panel['code'].nunique()}")
    print(panel.groupby("market")[["activations", "blended_cac", "organic_share"]]
          .mean().round(2).to_string())
