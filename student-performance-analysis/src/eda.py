"""Statistical EDA: data inspection, outliers, univariate / bivariate / multivariate statistics,
distribution fitting. Writes CSV tables and a results.json used by the report."""
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from config import EDA_DIR, TARGET, BINARY_COLS, NOMINAL_COLS, SEED
from data_preprocessing import load_data, clean_data, numeric_cols


def inspect(df):
    """Shape, dtypes, missing, duplicates, descriptive stats (mean/median/mode/std/quantiles)."""
    num = numeric_cols(df)
    dt = pd.DataFrame({"dtype": df.dtypes.astype(str), "n_unique": df.nunique(),
                       "missing": df.isna().sum()})
    dt.to_csv(EDA_DIR / "table_dtypes.csv")
    rows = []
    for c in num:
        s = df[c]
        rows.append({"feature": c, "mean": s.mean(), "median": s.median(),
                     "mode": s.mode().iloc[0], "std": s.std(), "min": s.min(),
                     "q05": s.quantile(.05), "q25": s.quantile(.25), "q75": s.quantile(.75),
                     "q95": s.quantile(.95), "max": s.max(), "skew": s.skew(),
                     "kurtosis": s.kurt()})
    desc = pd.DataFrame(rows).set_index("feature")
    desc.round(3).to_csv(EDA_DIR / "table_descriptive_numeric.csv")
    return dt, desc


def outlier_table(df):
    """IQR (1.5x) and |z|>3 outlier counts for numeric features with more than 10 distinct values."""
    rows = []
    for c in numeric_cols(df):
        s = df[c]
        if s.nunique() <= 10:
            continue
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_iqr = int(((s < lo) | (s > hi)).sum())
        n_z = int((np.abs((s - s.mean()) / s.std()) > 3).sum())
        rows.append({"feature": c, "lower_fence": lo, "upper_fence": hi,
                     "iqr_outliers": n_iqr, "iqr_pct": 100 * n_iqr / len(s), "z_gt_3": n_z})
    t = pd.DataFrame(rows).set_index("feature").sort_values("iqr_pct", ascending=False)
    t.round(3).to_csv(EDA_DIR / "table_outliers.csv")
    return t


def bivariate_tests(df):
    """Association of every feature with the 3-class target.
    numeric -> Kruskal-Wallis H (+ epsilon^2 effect size); binary/nominal -> chi-square (+ Cramer's V)."""
    n, k = len(df), df[TARGET].nunique()
    rows = []
    for c in numeric_cols(df):
        groups = [g[c].values for _, g in df.groupby(TARGET)]
        H, p = stats.kruskal(*groups)
        rows.append({"feature": c, "type": "numeric", "test": "Kruskal-Wallis",
                     "statistic": H, "p_value": p, "effect_size": (H - k + 1) / (n - k)})
    for c in BINARY_COLS + NOMINAL_COLS:
        ct = pd.crosstab(df[c], df[TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        rows.append({"feature": c, "type": "binary" if c in BINARY_COLS else "nominal",
                     "test": "Chi-square", "statistic": chi2, "p_value": p, "effect_size": v})
    t = pd.DataFrame(rows).set_index("feature").sort_values("effect_size", ascending=False)
    t.to_csv(EDA_DIR / "table_bivariate_tests.csv")
    return t


def multivariate(df):
    """Spearman correlation matrix, VIF (multicollinearity) and PCA on numeric features."""
    num = numeric_cols(df)
    corr = df[num].corr(method="spearman")
    corr.round(3).to_csv(EDA_DIR / "table_spearman_corr.csv")
    X = StandardScaler().fit_transform(df[num])
    vif = {}
    for i, c in enumerate(num):
        r2 = LinearRegression().fit(np.delete(X, i, 1), X[:, i]).score(np.delete(X, i, 1), X[:, i])
        vif[c] = 1 / (1 - r2) if r2 < 1 else np.inf
    vif = pd.Series(vif, name="VIF").sort_values(ascending=False)
    vif.round(2).to_csv(EDA_DIR / "table_vif.csv")
    pca = PCA().fit(X)
    return corr, vif, pca.explained_variance_ratio_


def distribution_fits(df):
    """Continuous (normal / log-normal) and discrete (Bernoulli-Binomial / Poisson) fits."""
    out = {}
    # --- continuous: normality of admission grade; log-normal vs normal for age
    for c in ["Admission grade", "Previous qualification (grade)"]:
        s = df[c]; mu, sd = stats.norm.fit(s)
        k2, p_norm = stats.normaltest(s)
        out[c] = {"mean": mu, "std": sd, "skew": s.skew(), "excess_kurtosis": s.kurt(),
                  "dagostino_p": p_norm, "ks_stat": stats.kstest(s, stats.norm(mu, sd).cdf).statistic}
    age = df["Age at enrollment"]
    shape, loc, scale = stats.lognorm.fit(age, floc=0)
    mu, sd = stats.norm.fit(age)
    out["Age at enrollment"] = {
        "skew": age.skew(),
        "ks_normal": stats.kstest(age, stats.norm(mu, sd).cdf).statistic,
        "ks_lognormal": stats.kstest(age, stats.lognorm(shape, loc, scale).cdf).statistic}
    # --- discrete: Bernoulli/Binomial for dropout; Poisson for count variables
    p_drop = (df[TARGET] == "Dropout").mean()
    out["dropout_proportion"] = p_drop
    lo, hi = stats.binomtest(int((df[TARGET] == "Dropout").sum()), len(df)).proportion_ci(method="wilson")
    out["dropout_wilson_ci"] = [lo, hi]
    for c in ["Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)",
              "Curricular units 1st sem (without evaluations)"]:
        s = df[c]
        out[c] = {"mean": s.mean(), "variance": s.var(), "dispersion_index": s.var() / s.mean(),
                  "zero_share": float((s == 0).mean())}
    return out


def run():
    raw = load_data()
    df, clean_report = clean_data(raw)
    dt, desc = inspect(df)
    out = {"shape": list(raw.shape), "clean_report": clean_report,
           "class_counts": df[TARGET].value_counts().to_dict(),
           "class_pct": (100 * df[TARGET].value_counts(normalize=True)).round(2).to_dict(),
           "n_numeric": len(numeric_cols(df)), "n_binary": len(BINARY_COLS),
           "n_nominal": len(NOMINAL_COLS), "dtypes": dt["dtype"].value_counts().to_dict()}
    ot = outlier_table(df); out["outliers_top"] = ot.head(6).round(2).reset_index().to_dict("records")
    bt = bivariate_tests(df); out["bivariate_top"] = bt.head(12).reset_index().round(4).to_dict("records")
    corr, vif, evr = multivariate(df)
    out["vif_top"] = vif.head(6).round(2).to_dict()
    out["pca_var_ratio_first5"] = [round(float(x), 4) for x in evr[:5]]
    out["distribution_fits"] = distribution_fits(df)
    # dropout rates for key binary features (descriptive insight)
    rates = {}
    for c in ["Tuition fees up to date", "Debtor", "Scholarship holder", "Gender", "Displaced"]:
        rates[c] = (df.groupby(c)[TARGET].apply(lambda s: round(100 * (s == "Dropout").mean(), 2))).to_dict()
    out["dropout_rate_by_binary"] = {k: {str(a): b for a, b in v.items()} for k, v in rates.items()}
    # strongest numeric correlations with each other
    c = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(key=abs, ascending=False)
    out["top_corr_pairs"] = [[a, b, round(float(v), 3)] for (a, b), v in c.head(6).items()]
    for f in ["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]:
        out.setdefault("grade_means_by_target", {})[f] = df.groupby(TARGET)[f].mean().round(2).to_dict()
    for f in ["Curricular units 2nd sem (approved)", "Age at enrollment", "Admission grade"]:
        out["grade_means_by_target"][f] = df.groupby(TARGET)[f].mean().round(2).to_dict()
    with open(EDA_DIR / "results_eda.json", "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(out, indent=1, default=str)[:3500])
    return df


if __name__ == "__main__":
    run()
