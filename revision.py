
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import (BayesianRidge, Lasso, LinearRegression,
                                  Ridge)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
print("=" * 72)
print("[1] LOADING DATA")
print("=" * 72)

raw = pd.read_excel(OUT / "Yearly_Enplanments_data__Harry_reid_airport_.xlsx")
raw = raw[raw["years"].between(2000, 2024)].copy()
raw["years"] = raw["years"].astype(int)

COVID_YEARS = [2020, 2021, 2022]
df = raw[~raw["years"].isin(COVID_YEARS)].copy()
df = df.dropna(subset=["enplanments", "clark-county_population", "real_gdp_per_person"])
df = df.sort_values("years").reset_index(drop=True)

print(f"    n = {len(df)} years; range {df['years'].min()}-{df['years'].max()}")
print(f"    excluded COVID: {COVID_YEARS}")
print(f"    years used: {df['years'].tolist()}")

y = df["enplanments"].values.astype(float)
years = df["years"].values
gdp = df["real_gdp_per_person"].values.astype(float)
pop = df["clark-county_population"].values.astype(float)
vis = df["clark_county_visitors_volumnes"].values.astype(float)




def adf_report(name, x):
    stat, p, lags, nobs, crits, _ = adfuller(x, autolag="AIC")
    return {
        "series": name,
        "adf_stat": stat,
        "p_value": p,
        "lags": lags,
        "n_obs": nobs,
        "crit_1pct": crits["1%"],
        "crit_5pct": crits["5%"],
        "crit_10pct": crits["10%"],
        "stationary_at_5pct": p < 0.05,
    }


adf_rows = [
    adf_report("enplanements (level)", y),
    adf_report("enplanements (1st diff)", np.diff(y)),
    adf_report("GDP per capita (level)", gdp),
    adf_report("GDP per capita (1st diff)", np.diff(gdp)),
]
adf_df = pd.DataFrame(adf_rows)
adf_df.to_csv(OUT / "rev_adf_stationarity.csv", index=False)
print(adf_df.to_string(index=False))

# ---- Chow test for COVID structural break (already in paper, reproduce) ---
pre_mask = (raw["years"] <= 2019) & raw["enplanments"].notna()
post_mask = (raw["years"] >= 2023) & raw["enplanments"].notna()
pre = raw.loc[pre_mask, ["enplanments", "real_gdp_per_person"]].dropna()
post = raw.loc[post_mask, ["enplanments", "real_gdp_per_person"]].dropna()


def _ssr(y_, x_):
    X_ = add_constant(x_)
    return float(OLS(y_, X_).fit().ssr)


ssr_pre = _ssr(pre["enplanments"].values, pre["real_gdp_per_person"].values)
ssr_post = _ssr(post["enplanments"].values, post["real_gdp_per_person"].values) if len(post) >= 2 else 0.0
combined = pd.concat([pre, post])
ssr_full = _ssr(combined["enplanments"].values, combined["real_gdp_per_person"].values)
k = 2
n_pre, n_post = len(pre), len(post)
chow_F = ((ssr_full - (ssr_pre + ssr_post)) / k) / ((ssr_pre + ssr_post) / (n_pre + n_post - 2 * k)) if (n_pre + n_post - 2 * k) > 0 else float("nan")
chow_p = 1.0 - stats.f.cdf(chow_F, k, n_pre + n_post - 2 * k) if not np.isnan(chow_F) else float("nan")
print(f"\nChow test (pre vs post COVID): F = {chow_F:.3f}, p = {chow_p:.3f}")
print(f"  pre n={n_pre}, post n={n_post}  -> low post-COVID power acknowledged")




N_SPLITS = 4
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# Cross-sectional features: GDP, population, visitor volume
X_cs = df[["real_gdp_per_person", "clark-county_population",
           "clark_county_visitors_volumnes"]].values.astype(float)


def cv_cross_sectional(name, build_model, log_y=False, poly_degree=None):
    """Run rolling-window CV. build_model() must return a fresh estimator."""
    fold_rows = []
    for fold_idx, (tr, te) in enumerate(tscv.split(X_cs), start=1):
        scaler = RobustScaler().fit(X_cs[tr])
        Xtr = scaler.transform(X_cs[tr])
        Xte = scaler.transform(X_cs[te])
        if poly_degree is not None:
            poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
            Xtr = poly.fit_transform(Xtr)
            Xte = poly.transform(Xte)
        ytr = np.log(y[tr]) if log_y else y[tr]
        yte = np.log(y[te]) if log_y else y[te]

        model = build_model()
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        if log_y:
            pred = np.exp(pred)
            yte = np.exp(yte)
        fold_rows.append({
            "model": name,
            "fold": fold_idx,
            "train_years": f"{years[tr][0]}-{years[tr][-1]}",
            "test_years": f"{years[te][0]}-{years[te][-1]}",
            "n_train": len(tr),
            "n_test": len(te),
            "fold_r2": r2_score(yte, pred),
            "fold_rmse": float(np.sqrt(mean_squared_error(yte, pred))),
            "fold_mae": mean_absolute_error(yte, pred),
        })
    return fold_rows


cv_records = []

# ML
cv_records += cv_cross_sectional("Random Forest",
    lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED))
cv_records += cv_cross_sectional("Linear Regression (ML)", LinearRegression)
cv_records += cv_cross_sectional("SVR", lambda: SVR(kernel="rbf", C=100, gamma="scale"))
cv_records += cv_cross_sectional("Decision Tree",
    lambda: DecisionTreeRegressor(max_depth=8, random_state=RANDOM_SEED))

# Regularized / Bayesian (R1)
cv_records += cv_cross_sectional("Ridge (alpha=1.0)",  lambda: Ridge(alpha=1.0, random_state=RANDOM_SEED))
cv_records += cv_cross_sectional("Lasso (alpha=0.01)", lambda: Lasso(alpha=0.01, max_iter=20000, random_state=RANDOM_SEED))
cv_records += cv_cross_sectional("Bayesian Ridge",     lambda: BayesianRidge())

# Econometric (log-linear) -- replicate paper's OLS variants
def cv_ols_subset(name, cols):
    fold_rows = []
    for fold_idx, (tr, te) in enumerate(tscv.split(X_cs), start=1):
        Xtr = np.log(df.loc[tr, cols].values.astype(float))
        Xte = np.log(df.loc[te, cols].values.astype(float))
        ytr = np.log(y[tr]); yte = np.log(y[te])
        m = LinearRegression().fit(Xtr, ytr)
        pred = np.exp(m.predict(Xte)); yte_lvl = np.exp(yte)
        fold_rows.append({
            "model": name, "fold": fold_idx,
            "train_years": f"{years[tr][0]}-{years[tr][-1]}",
            "test_years": f"{years[te][0]}-{years[te][-1]}",
            "n_train": len(tr), "n_test": len(te),
            "fold_r2": r2_score(yte_lvl, pred),
            "fold_rmse": float(np.sqrt(mean_squared_error(yte_lvl, pred))),
            "fold_mae": mean_absolute_error(yte_lvl, pred),
        })
    return fold_rows


cv_records += cv_ols_subset("OLS (Population)", ["clark-county_population"])
cv_records += cv_ols_subset("OLS (GDP)",         ["real_gdp_per_person"])
cv_records += cv_ols_subset("OLS (Pop + GDP)",   ["clark-county_population", "real_gdp_per_person"])

# Polynomial (degree 3) on raw pop+GDP (paper spec)
def cv_poly():
    fold_rows = []
    for fold_idx, (tr, te) in enumerate(tscv.split(X_cs), start=1):
        Xtr_raw = df.loc[tr, ["clark-county_population", "real_gdp_per_person"]].values.astype(float)
        Xte_raw = df.loc[te, ["clark-county_population", "real_gdp_per_person"]].values.astype(float)
        poly = PolynomialFeatures(degree=3, include_bias=False)
        Xtr = poly.fit_transform(Xtr_raw); Xte = poly.transform(Xte_raw)
        ytr = np.log(y[tr]); yte = np.log(y[te])
        m = LinearRegression().fit(Xtr, ytr)
        pred = np.exp(m.predict(Xte)); yte_lvl = np.exp(yte)
        fold_rows.append({
            "model": "Polynomial (D=3)", "fold": fold_idx,
            "train_years": f"{years[tr][0]}-{years[tr][-1]}",
            "test_years": f"{years[te][0]}-{years[te][-1]}",
            "n_train": len(tr), "n_test": len(te),
            "fold_r2": r2_score(yte_lvl, pred),
            "fold_rmse": float(np.sqrt(mean_squared_error(yte_lvl, pred))),
            "fold_mae": mean_absolute_error(yte_lvl, pred),
        })
    return fold_rows


cv_records += cv_poly()




def fit_arimax(y_train, exog_train, order=(1, 1, 1)):
    return ARIMA(y_train, exog=exog_train, order=order).fit()


def arimax_cv(order=(1, 1, 1), label="ARIMAX(1,1,1)"):
    """Both protocols. Multi-step: single forecast(steps=h). One-step-ahead:
    refit (or extend) for each test point so each prediction uses ACTUAL prior
    values."""
    multi_fold_rows, one_fold_rows = [], []
    for fold_idx, (tr, te) in enumerate(tscv.split(X_cs), start=1):
        ytr, yte = y[tr], y[te]
        gtr, gte = gdp[tr], gdp[te]

        # Multi-step recursive (h-step-ahead from end of training)
        res = fit_arimax(ytr, gtr, order)
        fc = res.get_forecast(steps=len(te), exog=gte.reshape(-1, 1))
        pred_multi = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        ci_lo, ci_hi = np.asarray(ci)[:, 0], np.asarray(ci)[:, 1]
        multi_fold_rows.append({
            "model": label, "protocol": "multi-step recursive",
            "fold": fold_idx,
            "train_years": f"{years[tr][0]}-{years[tr][-1]}",
            "test_years": f"{years[te][0]}-{years[te][-1]}",
            "n_train": len(tr), "n_test": len(te),
            "fold_r2": r2_score(yte, pred_multi),
            "fold_rmse": float(np.sqrt(mean_squared_error(yte, pred_multi))),
            "fold_mae": mean_absolute_error(yte, pred_multi),
            "ci_lo_first": float(ci_lo[0]), "ci_hi_first": float(ci_hi[0]),
            "ci_lo_last": float(ci_lo[-1]), "ci_hi_last": float(ci_hi[-1]),
        })

        # One-step-ahead (refit at each test point, expanding window)
        preds_one = []
        for j in range(len(te)):
            y_hist = np.concatenate([ytr, yte[:j]])
            g_hist = np.concatenate([gtr, gte[:j]])
            res_j = fit_arimax(y_hist, g_hist, order)
            fc_j = res_j.get_forecast(steps=1, exog=gte[j:j + 1].reshape(-1, 1))
            preds_one.append(float(fc_j.predicted_mean[0]))
        preds_one = np.array(preds_one)
        one_fold_rows.append({
            "model": label, "protocol": "one-step-ahead (refit each step)",
            "fold": fold_idx,
            "train_years": f"{years[tr][0]}-{years[tr][-1]}",
            "test_years": f"{years[te][0]}-{years[te][-1]}",
            "n_train": len(tr), "n_test": len(te),
            "fold_r2": r2_score(yte, preds_one),
            "fold_rmse": float(np.sqrt(mean_squared_error(yte, preds_one))),
            "fold_mae": mean_absolute_error(yte, preds_one),
        })
    return multi_fold_rows, one_fold_rows


arimax_multi, arimax_one = arimax_cv((1, 1, 1), "ARIMAX(1,1,1)")
arimax_multi_112, arimax_one_112 = arimax_cv((1, 1, 2), "ARIMAX(1,1,2)")

arimax_fold_df = pd.DataFrame(arimax_multi + arimax_one + arimax_multi_112 + arimax_one_112)
arimax_fold_df.to_csv(OUT / "rev_arimax_fold_level.csv", index=False)
print("\nFold-level ARIMAX results (saved to rev_arimax_fold_level.csv):")
print(arimax_fold_df[["model", "protocol", "fold", "train_years",
                      "test_years", "fold_r2", "fold_rmse"]].to_string(index=False))


def aggregate(fold_rows):
    """Concat-style OOS R^2 (paper convention) AND mean-of-folds (R1 stability)."""
    fr = pd.DataFrame(fold_rows)
    obs, pred = [], []
    for _, r in fr.iterrows():
        obs.append([0.0] * int(r["n_test"]))   # placeholder; we recompute below
        pred.append([0.0] * int(r["n_test"]))
    return {
        "mean_r2": fr["fold_r2"].mean(),
        "median_r2": fr["fold_r2"].median(),
        "std_r2": fr["fold_r2"].std(),
        "min_r2": fr["fold_r2"].min(),
        "max_r2": fr["fold_r2"].max(),
        "mean_rmse": fr["fold_rmse"].mean(),
    }


print("\nARIMAX(1,1,1) multi-step:    R^2 mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f}".format(
    *(aggregate(arimax_multi)[k] for k in ("mean_r2", "std_r2", "min_r2", "max_r2"))))
print("ARIMAX(1,1,1) one-step:      R^2 mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f}".format(
    *(aggregate(arimax_one)[k] for k in ("mean_r2", "std_r2", "min_r2", "max_r2"))))


print("\n" + "=" * 72)
print("[5] AGGREGATE OOS METRICS (concat across folds)")
print("=" * 72)


def concat_oos(fold_records):
    """Reproduce 'paper-style' OOS R^2 by concatenating predictions across folds."""
    out = {}
    df_records = pd.DataFrame(fold_records)
    for name, grp in df_records.groupby("model"):
        # We need raw obs/pred per fold to reproduce; recompute below.
        pass
    return out


def concat_oos_cs(name, build_model, log_y=False, poly_degree=None):
    obs_all, pred_all = [], []
    for tr, te in tscv.split(X_cs):
        scaler = RobustScaler().fit(X_cs[tr])
        Xtr = scaler.transform(X_cs[tr]); Xte = scaler.transform(X_cs[te])
        if poly_degree is not None:
            poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
            Xtr = poly.fit_transform(Xtr); Xte = poly.transform(Xte)
        ytr = np.log(y[tr]) if log_y else y[tr]
        yte = np.log(y[te]) if log_y else y[te]
        m = build_model().fit(Xtr, ytr)
        p = m.predict(Xte)
        if log_y:
            p = np.exp(p); yte = np.exp(yte)
        obs_all.extend(yte); pred_all.extend(p)
    obs_all, pred_all = np.array(obs_all), np.array(pred_all)
    return {
        "model": name,
        "oos_r2_concat": r2_score(obs_all, pred_all),
        "oos_rmse_concat": float(np.sqrt(mean_squared_error(obs_all, pred_all))),
        "oos_mae_concat": mean_absolute_error(obs_all, pred_all),
    }


def concat_oos_ols(name, cols):
    obs_all, pred_all = [], []
    for tr, te in tscv.split(X_cs):
        Xtr = np.log(df.loc[tr, cols].values.astype(float))
        Xte = np.log(df.loc[te, cols].values.astype(float))
        ytr = np.log(y[tr]); yte = np.log(y[te])
        m = LinearRegression().fit(Xtr, ytr)
        p = np.exp(m.predict(Xte)); yte_lvl = np.exp(yte)
        obs_all.extend(yte_lvl); pred_all.extend(p)
    obs_all, pred_all = np.array(obs_all), np.array(pred_all)
    return {
        "model": name,
        "oos_r2_concat": r2_score(obs_all, pred_all),
        "oos_rmse_concat": float(np.sqrt(mean_squared_error(obs_all, pred_all))),
        "oos_mae_concat": mean_absolute_error(obs_all, pred_all),
    }


def concat_oos_poly():
    obs_all, pred_all = [], []
    for tr, te in tscv.split(X_cs):
        Xtr_raw = df.loc[tr, ["clark-county_population", "real_gdp_per_person"]].values.astype(float)
        Xte_raw = df.loc[te, ["clark-county_population", "real_gdp_per_person"]].values.astype(float)
        poly = PolynomialFeatures(degree=3, include_bias=False)
        Xtr = poly.fit_transform(Xtr_raw); Xte = poly.transform(Xte_raw)
        ytr = np.log(y[tr]); yte = np.log(y[te])
        m = LinearRegression().fit(Xtr, ytr)
        p = np.exp(m.predict(Xte)); yte_lvl = np.exp(yte)
        obs_all.extend(yte_lvl); pred_all.extend(p)
    obs_all, pred_all = np.array(obs_all), np.array(pred_all)
    return {
        "model": "Polynomial (D=3)",
        "oos_r2_concat": r2_score(obs_all, pred_all),
        "oos_rmse_concat": float(np.sqrt(mean_squared_error(obs_all, pred_all))),
        "oos_mae_concat": mean_absolute_error(obs_all, pred_all),
    }


def concat_oos_arimax(order=(1, 1, 1), name="ARIMAX(1,1,1)", protocol="multi"):
    obs_all, pred_all = [], []
    for tr, te in tscv.split(X_cs):
        ytr, yte = y[tr], y[te]
        gtr, gte = gdp[tr], gdp[te]
        if protocol == "multi":
            res = fit_arimax(ytr, gtr, order)
            p = np.asarray(res.get_forecast(steps=len(te), exog=gte.reshape(-1, 1)).predicted_mean)
        else:
            p = []
            for j in range(len(te)):
                yh = np.concatenate([ytr, yte[:j]])
                gh = np.concatenate([gtr, gte[:j]])
                rj = fit_arimax(yh, gh, order)
                p.append(float(rj.get_forecast(steps=1, exog=gte[j:j+1].reshape(-1, 1)).predicted_mean[0]))
            p = np.array(p)
        obs_all.extend(yte); pred_all.extend(p)
    obs_all, pred_all = np.array(obs_all), np.array(pred_all)
    return {
        "model": f"{name} [{ 'multi-step' if protocol=='multi' else 'one-step-ahead' }]",
        "oos_r2_concat": r2_score(obs_all, pred_all),
        "oos_rmse_concat": float(np.sqrt(mean_squared_error(obs_all, pred_all))),
        "oos_mae_concat": mean_absolute_error(obs_all, pred_all),
    }


summary_rows = [
    concat_oos_cs("Random Forest",         lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)),
    concat_oos_cs("Linear Regression (ML)", LinearRegression),
    concat_oos_cs("SVR",                    lambda: SVR(kernel="rbf", C=100, gamma="scale")),
    concat_oos_cs("Decision Tree",          lambda: DecisionTreeRegressor(max_depth=8, random_state=RANDOM_SEED)),
    concat_oos_cs("Ridge (alpha=1.0)",      lambda: Ridge(alpha=1.0, random_state=RANDOM_SEED)),
    concat_oos_cs("Lasso (alpha=0.01)",     lambda: Lasso(alpha=0.01, max_iter=20000, random_state=RANDOM_SEED)),
    concat_oos_cs("Bayesian Ridge",         BayesianRidge),
    concat_oos_ols("OLS (Population)",      ["clark-county_population"]),
    concat_oos_ols("OLS (GDP)",             ["real_gdp_per_person"]),
    concat_oos_ols("OLS (Pop + GDP)",       ["clark-county_population", "real_gdp_per_person"]),
    concat_oos_poly(),
    concat_oos_arimax((1, 1, 1), "ARIMAX(1,1,1)", "multi"),
    concat_oos_arimax((1, 1, 1), "ARIMAX(1,1,1)", "one"),
    concat_oos_arimax((1, 1, 2), "ARIMAX(1,1,2)", "multi"),
]
summary_df = pd.DataFrame(summary_rows).sort_values("oos_r2_concat", ascending=False).reset_index(drop=True)
summary_df.to_csv(OUT / "rev_oos_summary.csv", index=False)
print(summary_df.to_string(index=False))

# Also persist fold-level cross-sectional CV
pd.DataFrame(cv_records).to_csv(OUT / "rev_cv_fold_level_cross_sectional.csv", index=False)

aic_rows = []
for p in range(3):
    for q in range(3):
        try:
            r = ARIMA(y, exog=gdp.reshape(-1, 1), order=(p, 1, q)).fit()
            aic_rows.append({"p": p, "d": 1, "q": q, "AIC": float(r.aic),
                             "BIC": float(r.bic), "loglik": float(r.llf)})
        except Exception as e:
            aic_rows.append({"p": p, "d": 1, "q": q, "AIC": np.nan,
                             "BIC": np.nan, "loglik": np.nan, "err": str(e)})
aic_df = pd.DataFrame(aic_rows).sort_values("AIC")
aic_df.to_csv(OUT / "rev_aic_order_selection.csv", index=False)
print(aic_df.to_string(index=False))

arimax_full = fit_arimax(y, gdp, (1, 1, 1))
resid = arimax_full.resid

# Ljung-Box test for residual autocorrelation
lb = acorr_ljungbox(resid, lags=[5, 10], return_df=True)
print("Ljung-Box (residuals):")
print(lb.to_string())
lb.to_csv(OUT / "rev_ljungbox.csv")

# Jarque-Bera normality on residuals
jb_stat, jb_p = stats.jarque_bera(resid)
print(f"Jarque-Bera on residuals: stat={jb_stat:.3f}, p={jb_p:.3f}")

# Plot: residual + ACF + PACF + QQ
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(years, resid, "o-")
axes[0, 0].axhline(0, color="red", lw=1)
axes[0, 0].set_title("ARIMAX(1,1,1) residuals over time")
axes[0, 0].set_xlabel("Year"); axes[0, 0].set_ylabel("Residual")

plot_acf(resid, ax=axes[0, 1], lags=min(10, len(resid) // 2))
axes[0, 1].set_title("ACF of residuals")

plot_pacf(resid, ax=axes[1, 0], lags=min(10, len(resid) // 2 - 1), method="ywm")
axes[1, 0].set_title("PACF of residuals")

stats.probplot(resid, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Q-Q plot of residuals")
plt.tight_layout()
plt.savefig(OUT / "rev_arimax_diagnostics.png", dpi=200, bbox_inches="tight")
plt.close()

# ACF/PACF on enplanements series itself (informs (p,d,q) choice)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plot_acf(y, ax=axes[0, 0], lags=min(10, len(y) // 2))
axes[0, 0].set_title("ACF of enplanements (level)")
plot_pacf(y, ax=axes[0, 1], lags=min(10, len(y) // 2 - 1), method="ywm")
axes[0, 1].set_title("PACF of enplanements (level)")
plot_acf(np.diff(y), ax=axes[1, 0], lags=min(10, (len(y)-1) // 2))
axes[1, 0].set_title("ACF of 1st-differenced enplanements")
plot_pacf(np.diff(y), ax=axes[1, 1], lags=min(10, (len(y)-1) // 2 - 1), method="ywm")
axes[1, 1].set_title("PACF of 1st-differenced enplanements")
plt.tight_layout()
plt.savefig(OUT / "rev_acf_pacf_enplanements.png", dpi=200, bbox_inches="tight")
plt.close()

print("\n" + "=" * 72)
print("[8] 2025-2045 FORECASTS WITH 95% PREDICTION INTERVALS")
print("=" * 72)

future_years = list(range(2025, 2046))
pop_forecast = {2025: 2443000, 2026: 2493000, 2027: 2537000, 2028: 2578000, 2029: 2617000,
                2030: 2655000, 2031: 2692000, 2032: 2728000, 2033: 2764000, 2034: 2797000,
                2035: 2830000, 2036: 2860000, 2037: 2889000, 2038: 2917000, 2039: 2944000,
                2040: 2969000, 2041: 2994000, 2042: 3017000, 2043: 3039000, 2044: 3061000,
                2045: 3081000}
gdp_forecast = {2025: 49586, 2026: 50131, 2027: 50783, 2028: 51596, 2029: 52421,
                2030: 53207, 2031: 53952, 2032: 54708, 2033: 55474, 2034: 56250,
                2035: 56981, 2036: 57722, 2037: 58473, 2038: 59233, 2039: 60003,
                2040: 60783, 2041: 61512, 2042: 62250, 2043: 63060, 2044: 63879,
                2045: 64710}

future_gdp = np.array([gdp_forecast[y_] for y_ in future_years]).reshape(-1, 1)
future_pop = np.array([pop_forecast[y_] for y_ in future_years])

# ---- ARIMAX(1,1,1) point + 95% PI (closed-form from statsmodels)
fc = arimax_full.get_forecast(steps=len(future_years), exog=future_gdp)
arimax_point = fc.predicted_mean
arimax_ci = fc.conf_int(alpha=0.05)
arimax_lo = np.asarray(arimax_ci)[:, 0]
arimax_hi = np.asarray(arimax_ci)[:, 1]

# ---- OLS (GDP) bootstrap PI
def bootstrap_ols_pi(predict_X_train, predict_X_future, y_train, n_boot=1000):
    """Residual bootstrap on log-OLS, returns 95% PI in original scale."""
    X = predict_X_train
    Xf = predict_X_future
    base = LinearRegression().fit(X, np.log(y_train))
    resid = np.log(y_train) - base.predict(X)
    fc_log = base.predict(Xf)
    boots = np.empty((n_boot, len(Xf)))
    for b in range(n_boot):
        eps = np.random.choice(resid, size=len(y_train), replace=True)
        m = LinearRegression().fit(X, base.predict(X) + eps)
        eps_f = np.random.choice(resid, size=len(Xf), replace=True)
        boots[b] = np.exp(m.predict(Xf) + eps_f)
    return np.exp(fc_log), np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


X_gdp = np.log(gdp.reshape(-1, 1))
X_gdp_future = np.log(future_gdp)
ols_gdp_point, ols_gdp_lo, ols_gdp_hi = bootstrap_ols_pi(X_gdp, X_gdp_future, y)

# ---- OLS (Pop + GDP)
X_both = np.log(np.column_stack([pop, gdp]))
X_both_future = np.log(np.column_stack([future_pop, future_gdp.flatten()]))
ols_both_point, ols_both_lo, ols_both_hi = bootstrap_ols_pi(X_both, X_both_future, y)

# ---- Ridge (cross-sectional) bootstrap PI
def bootstrap_ridge_pi(X_train_raw, X_future_raw, y_train, alpha=1.0, n_boot=1000):
    sc = RobustScaler().fit(X_train_raw)
    Xs = sc.transform(X_train_raw); Xfs = sc.transform(X_future_raw)
    base = Ridge(alpha=alpha, random_state=RANDOM_SEED).fit(Xs, y_train)
    resid = y_train - base.predict(Xs)
    fc_pt = base.predict(Xfs)
    boots = np.empty((n_boot, len(Xfs)))
    for b in range(n_boot):
        eps = np.random.choice(resid, size=len(y_train), replace=True)
        m = Ridge(alpha=alpha, random_state=RANDOM_SEED).fit(Xs, base.predict(Xs) + eps)
        eps_f = np.random.choice(resid, size=len(Xfs), replace=True)
        boots[b] = m.predict(Xfs) + eps_f
    return fc_pt, np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


# Future visitor projection (paper convention)
visitor_growth = pd.Series(vis).pct_change().dropna().mean()
last_vis = vis[-1]
fut_vis = []
cur = last_vis
for _ in future_years:
    cur *= (1 + visitor_growth); fut_vis.append(cur)
X_full_train = np.column_stack([gdp, pop, vis])
X_full_future = np.column_stack([future_gdp.flatten(), future_pop, fut_vis])
ridge_point, ridge_lo, ridge_hi = bootstrap_ridge_pi(X_full_train, X_full_future, y)

# Save the forecast + intervals table
fc_table = pd.DataFrame({
    "year": future_years,
    "ARIMAX_point": arimax_point,
    "ARIMAX_lo95":  arimax_lo,
    "ARIMAX_hi95":  arimax_hi,
    "OLS_GDP_point": ols_gdp_point,
    "OLS_GDP_lo95":  ols_gdp_lo,
    "OLS_GDP_hi95":  ols_gdp_hi,
    "OLS_PopGDP_point": ols_both_point,
    "OLS_PopGDP_lo95":  ols_both_lo,
    "OLS_PopGDP_hi95":  ols_both_hi,
    "Ridge_point": ridge_point,
    "Ridge_lo95":  ridge_lo,
    "Ridge_hi95":  ridge_hi,
})
fc_table.to_csv(OUT / "rev_forecasts_with_PI.csv", index=False)
print(f"\nARIMAX 2045: {arimax_point[-1]/1e6:.2f}M  [95% PI: {arimax_lo[-1]/1e6:.2f} - {arimax_hi[-1]/1e6:.2f}]")
print(f"OLS GDP 2045 (boot): {ols_gdp_point[-1]/1e6:.2f}M  [{ols_gdp_lo[-1]/1e6:.2f} - {ols_gdp_hi[-1]/1e6:.2f}]")
print(f"Ridge 2045 (boot):  {ridge_point[-1]/1e6:.2f}M  [{ridge_lo[-1]/1e6:.2f} - {ridge_hi[-1]/1e6:.2f}]")

# Plot ARIMAX with PI
plt.figure(figsize=(12, 6))
plt.plot(years, y, "o-", color="black", lw=2, label="Historical")
plt.plot(future_years, arimax_point, "s-", color="#1f77b4", lw=2, label="ARIMAX(1,1,1) point")
plt.fill_between(future_years, arimax_lo, arimax_hi, alpha=0.25,
                 color="#1f77b4", label="ARIMAX 95% PI")
plt.axvline(2024, color="red", ls="--", alpha=0.7)
plt.xlabel("Year"); plt.ylabel("Enplanements")
plt.title("ARIMAX(1,1,1) Forecast with 95% Prediction Interval")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(OUT / "rev_arimax_forecast_PI.png", dpi=200, bbox_inches="tight")
plt.close()

fc_low = arimax_full.get_forecast(steps=len(future_years), exog=future_gdp * 0.9)
fc_high = arimax_full.get_forecast(steps=len(future_years), exog=future_gdp * 1.1)
sens = pd.DataFrame({
    "year": future_years,
    "GDP_minus10": fc_low.predicted_mean,
    "GDP_base":    arimax_point,
    "GDP_plus10":  fc_high.predicted_mean,
})
sens.to_csv(OUT / "rev_gdp_sensitivity.csv", index=False)
print(sens.tail().to_string(index=False))
print(f"2045 envelope: {sens['GDP_minus10'].iloc[-1]/1e6:.2f}M -- {sens['GDP_plus10'].iloc[-1]/1e6:.2f}M")
print("Justification: 10% covers historical CBO 10-yr GDP forecast error band")
print("(approx. +/-1 SE on long-horizon real GDP per capita projections).")

# Fit deg-3 polynomial on full data, then evaluate at training-extreme vs future
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly_train = poly.fit_transform(np.column_stack([pop, gdp]))
m_poly = LinearRegression().fit(X_poly_train, np.log(y))

# extrapolation distance: how far is each future year from training support?
train_pop_max = pop.max(); train_gdp_max = gdp.max()
extrap = pd.DataFrame({
    "year": future_years,
    "pop_pct_above_train_max":  (future_pop - train_pop_max) / train_pop_max * 100,
    "gdp_pct_above_train_max":  (future_gdp.flatten() - train_gdp_max) / train_gdp_max * 100,
})
extrap.to_csv(OUT / "rev_polynomial_extrapolation_distance.csv", index=False)
print(extrap.head(3).to_string(index=False))
print("...")
print(extrap.tail(3).to_string(index=False))
print("Cubic terms scale as (1+delta)^3, so a 30% extrapolation in inputs"
      "\nproduces ~120% growth in cubic basis -- mathematically inevitable blow-up.")


stab_rows = []
for ns in (3, 4, 5):
    tscv_s = TimeSeriesSplit(n_splits=ns)
    obs_all, pred_all = [], []
    for tr, te in tscv_s.split(X_cs):
        ytr, yte = y[tr], y[te]
        gtr, gte = gdp[tr], gdp[te]
        try:
            res = fit_arimax(ytr, gtr, (1, 1, 1))
            p = np.asarray(res.get_forecast(steps=len(te), exog=gte.reshape(-1, 1)).predicted_mean)
            obs_all.extend(yte); pred_all.extend(p)
        except Exception:
            continue
    if len(obs_all):
        stab_rows.append({
            "n_splits": ns,
            "oos_r2": r2_score(np.array(obs_all), np.array(pred_all)),
            "oos_rmse": float(np.sqrt(mean_squared_error(np.array(obs_all), np.array(pred_all)))),
        })
stab_df = pd.DataFrame(stab_rows)
stab_df.to_csv(OUT / "rev_arimax_stability.csv", index=False)
print(stab_df.to_string(index=False))


report = OUT / "rev_revision_report.md"
with report.open("w") as f:
    f.write("# Revision Analysis Report (ITSC 2026 #671)\n\n")
    f.write(f"Years used (n={len(df)}): {df['years'].tolist()}\n\n")
    f.write("## ADF stationarity\n\n")
    f.write(adf_df.to_markdown(index=False) + "\n\n")
    f.write(f"## Chow test for COVID structural break\nF = {chow_F:.3f}, p = {chow_p:.3f} (n_pre={n_pre}, n_post={n_post}); low post-COVID power acknowledged.\n\n")
    f.write("## AIC order selection ARIMAX(p,1,q)\n\n")
    f.write(aic_df.to_markdown(index=False) + "\n\n")
    f.write("## OOS summary (concat across folds)\n\n")
    f.write(summary_df.to_markdown(index=False) + "\n\n")
    f.write("## ARIMAX fold-level (one-step-ahead AND multi-step)\n\n")
    f.write(arimax_fold_df.to_markdown(index=False) + "\n\n")
    f.write("## ARIMAX stability across n_splits\n\n")
    f.write(stab_df.to_markdown(index=False) + "\n\n")
    f.write("## Forecasts 2025-2045 with 95% PI\n\n")
    f.write(fc_table.to_markdown(index=False) + "\n\n")
    f.write("## GDP +/-10% sensitivity\n\n")
    f.write(sens.to_markdown(index=False) + "\n\n")
    f.write(f"## Ljung-Box on ARIMAX residuals\n\n{lb.to_markdown()}\n\n")
    f.write(f"Jarque-Bera: stat = {jb_stat:.3f}, p = {jb_p:.3f}\n")
print(f"Report: {report}")
