"""EDA visualisations (Figures 1-12) + two interactive Plotly pages (stage 7: Interact)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import (EDA_DIR, TARGET, CLASS_COLORS, set_style, save_fig, SEED,
                    BINARY_COLS, CLASS_NAMES)
from data_preprocessing import load_data, clean_data, numeric_cols

ORDER = ["Dropout", "Enrolled", "Graduate"]
PAL = {k: CLASS_COLORS[k] for k in ORDER}


def fig01_target(df):
    """Class balance of the target (bar chart with counts and %)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    cnt = df[TARGET].value_counts().reindex(ORDER)
    bars = ax.bar(cnt.index, cnt.values, color=[PAL[k] for k in cnt.index], width=0.55)
    for b, v in zip(bars, cnt.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 30, f"{v}\n({100*v/len(df):.1f}%)", ha="center", fontsize=9)
    ax.set(title="Fig. 1  Student outcome (target) distribution", ylabel="Number of students",
           xlabel="Outcome at end of normal course duration", ylim=(0, cnt.max() * 1.18))
    ax.grid(axis="x", visible=False)
    save_fig(fig, EDA_DIR / "fig01_target_distribution.png")


def fig02_hist(df):
    """Histograms with mean (solid) and median (dashed) for six key numeric variables."""
    cols = ["Admission grade", "Previous qualification (grade)", "Age at enrollment",
            "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)",
            "Curricular units 2nd sem (approved)"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, c in zip(axes.ravel(), cols):
        s = df[c]
        ax.hist(s, bins=30, color="#2a78d6", edgecolor="white", linewidth=0.5)
        ax.axvline(s.mean(), color="#0b0b0b", lw=1.6, label=f"mean {s.mean():.1f}")
        ax.axvline(s.median(), color="#eb6834", lw=1.6, ls="--", label=f"median {s.median():.1f}")
        ax.set(title=c, xlabel=c, ylabel="Frequency"); ax.legend(fontsize=8)
        ax.title.set_fontsize(10)
    fig.suptitle("Fig. 2  Histograms of key numeric variables", fontweight="bold", y=1.0)
    fig.tight_layout(); save_fig(fig, EDA_DIR / "fig02_histograms.png")


def fig03_box(df):
    """Boxplots of key variables split by outcome (outliers shown as points)."""
    cols = ["Admission grade", "Age at enrollment", "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (approved)"]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.8))
    for ax, c in zip(axes, cols):
        sns.boxplot(data=df, x=TARGET, y=c, order=ORDER, palette=PAL, hue=TARGET, legend=False,
                    width=0.6, linewidth=1, fliersize=2, ax=ax)
        ax.set_title(c, fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Fig. 3  Boxplots of key variables by student outcome", fontweight="bold", y=1.03)
    fig.tight_layout(); save_fig(fig, EDA_DIR / "fig03_boxplots_by_target.png")


def fig04_outliers(df):
    """Share of IQR outliers per feature (horizontal bars)."""
    t = pd.read_csv(EDA_DIR / "table_outliers.csv", index_col=0).sort_values("iqr_pct")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(t.index, t["iqr_pct"], color="#2a78d6", height=0.65)
    for y, v in enumerate(t["iqr_pct"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set(title="Fig. 4  Share of IQR outliers per numeric feature",
           xlabel="Rows outside Q1 - 1.5 IQR / Q3 + 1.5 IQR (%)"); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, t["iqr_pct"].max() * 1.15)
    save_fig(fig, EDA_DIR / "fig04_outlier_share.png")


def fig05_normal(df):
    """Continuous distribution: normal fit on admission grade + Q-Q plot."""
    s = df["Admission grade"]; mu, sd = stats.norm.fit(s)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9))
    a.hist(s, bins=40, density=True, color="#2a78d6", edgecolor="white", linewidth=0.4, label="Observed")
    x = np.linspace(s.min(), s.max(), 300)
    a.plot(x, stats.norm.pdf(x, mu, sd), color="#eb6834", lw=2, label=f"Normal fit (μ={mu:.1f}, σ={sd:.1f})")
    a.set(title="(a) Histogram vs fitted normal", xlabel="Admission grade", ylabel="Density"); a.legend()
    stats.probplot(s, dist="norm", plot=b)
    b.get_lines()[0].set(color="#2a78d6", markersize=3); b.get_lines()[1].set(color="#eb6834", lw=2)
    b.set(title="(b) Normal Q-Q plot", xlabel="Theoretical quantiles", ylabel="Ordered values")
    fig.suptitle("Fig. 5  Continuous distribution: Admission grade", fontweight="bold", y=1.03)
    fig.tight_layout(); save_fig(fig, EDA_DIR / "fig05_normal_fit_qq.png")


def fig06_discrete(df):
    """Discrete distributions: (a) Poisson fit to approved units, (b) Binomial(10, p) for dropouts."""
    s = df["Curricular units 1st sem (approved)"]; lam = s.mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    k = np.arange(0, s.max() + 1)
    obs = s.value_counts(normalize=True).reindex(k, fill_value=0)
    a.bar(k - 0.2, obs.values, width=0.4, color="#2a78d6", label="Observed")
    a.bar(k + 0.2, stats.poisson.pmf(k, lam), width=0.4, color="#eb6834", label=f"Poisson (λ={lam:.2f})")
    a.set(title="(a) Units approved, 1st semester", xlabel="Units approved", ylabel="Probability"); a.legend()
    p = (df[TARGET] == "Dropout").mean(); n = 10
    rng = np.random.default_rng(SEED)
    sims = np.array([(df[TARGET].sample(n, random_state=int(r)) == "Dropout").sum()
                     for r in rng.integers(0, 10**6, 5000)])
    kk = np.arange(0, n + 1)
    emp = np.bincount(sims, minlength=n + 1) / len(sims)
    b.bar(kk - 0.2, emp, width=0.4, color="#2a78d6", label="Simulated (5000 samples of 10 students)")
    b.bar(kk + 0.2, stats.binom.pmf(kk, n, p), width=0.4, color="#eb6834", label=f"Binomial(10, p={p:.3f})")
    b.set(title="(b) Dropouts among 10 random students", xlabel="Number of dropouts", ylabel="Probability")
    b.legend(fontsize=8)
    fig.suptitle("Fig. 6  Discrete distributions: Poisson and Binomial", fontweight="bold", y=1.03)
    fig.tight_layout(); save_fig(fig, EDA_DIR / "fig06_discrete_distributions.png")


def fig07_corr(df):
    """Spearman correlation heatmap of numeric features (diverging palette, neutral midpoint)."""
    num = numeric_cols(df); corr = df[num].corr(method="spearman")
    short = [c.replace("Curricular units ", "CU ").replace("Previous qualification", "Prev. qual.")
             .replace(" sem", "") for c in num]
    fig, ax = plt.subplots(figsize=(9, 7.4))
    mask = np.triu(np.ones_like(corr, bool), 1)
    sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(40, 250, s=85, l=50, as_cmap=True),
                center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 6},
                xticklabels=short, yticklabels=short, linewidths=0.4, linecolor="white",
                cbar_kws={"label": "Spearman ρ", "shrink": 0.7}, ax=ax)
    ax.grid(False); ax.set_title("Fig. 7  Spearman correlation heatmap (numeric features)")
    save_fig(fig, EDA_DIR / "fig07_correlation_heatmap.png")


def fig08_binary(df):
    """100% stacked bars: outcome mix across four binary risk/context factors."""
    cols = {"Tuition fees up to date": ("No (0)", "Yes (1)"), "Debtor": ("No (0)", "Yes (1)"),
            "Scholarship holder": ("No (0)", "Yes (1)"), "Gender": ("Code 0", "Code 1")}
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.8), sharey=True)
    for ax, (c, labs) in zip(axes, cols.items()):
        ct = pd.crosstab(df[c], df[TARGET], normalize="index").reindex(columns=ORDER) * 100
        bottom = np.zeros(len(ct))
        for cls in ORDER:
            ax.bar(range(len(ct)), ct[cls], bottom=bottom, color=PAL[cls], width=0.6,
                   edgecolor="white", linewidth=1.2, label=cls)
            for i, v in enumerate(ct[cls]):
                if v > 7: ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white")
            bottom += ct[cls].values
        ax.set_xticks(range(len(ct))); ax.set_xticklabels(labs); ax.set_title(c, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of students (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Fig. 8  Outcome mix by binary factors", fontweight="bold", y=1.02)
    fig.tight_layout(); save_fig(fig, EDA_DIR / "fig08_outcome_by_binary_factors.png")


def fig09_scatter(df):
    """Scatterplot of 1st vs 2nd semester grade coloured by outcome (jittered, semi-transparent)."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    rng = np.random.default_rng(SEED)
    for cls in ["Graduate", "Enrolled", "Dropout"]:
        d = df[df[TARGET] == cls]
        ax.scatter(d["Curricular units 1st sem (grade)"] + rng.normal(0, .08, len(d)),
                   d["Curricular units 2nd sem (grade)"] + rng.normal(0, .08, len(d)),
                   s=12, alpha=0.45, color=PAL[cls], label=cls, edgecolor="none")
    ax.set(title="Fig. 9  First vs second semester grade", xlabel="1st semester grade (0-20)",
           ylabel="2nd semester grade (0-20)")
    ax.legend(markerscale=2.2, title="Outcome")
    save_fig(fig, EDA_DIR / "fig09_grade_scatter.png")


def fig10_pca(df):
    """PCA projection of standardised numeric features coloured by outcome."""
    num = numeric_cols(df); Z = StandardScaler().fit_transform(df[num])
    pca = PCA(2, random_state=SEED).fit(Z); P = pca.transform(Z)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for cls in ["Graduate", "Enrolled", "Dropout"]:
        m = (df[TARGET] == cls).values
        ax.scatter(P[m, 0], P[m, 1], s=10, alpha=0.4, color=PAL[cls], label=cls, edgecolor="none")
    ev = pca.explained_variance_ratio_ * 100
    ax.set(title="Fig. 10  PCA projection of numeric features", xlabel=f"PC1 ({ev[0]:.1f}% variance)",
           ylabel=f"PC2 ({ev[1]:.1f}% variance)")
    ax.legend(markerscale=2.5, title="Outcome")
    save_fig(fig, EDA_DIR / "fig10_pca.png")


def fig11_pair(df):
    """Pairplot of five key numeric variables coloured by outcome."""
    cols = ["Admission grade", "Age at enrollment", "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (approved)"]
    short = {c: c.replace("Curricular units ", "CU ").replace(" sem", "") for c in cols}
    d = df[cols + [TARGET]].rename(columns=short)
    g = sns.pairplot(d, hue=TARGET, hue_order=ORDER, palette=PAL, corner=True,
                     plot_kws={"s": 8, "alpha": 0.35, "edgecolor": "none"}, diag_kind="kde", height=1.9)
    g.figure.suptitle("Fig. 11  Pairplot of key variables", fontweight="bold", y=1.01)
    g.figure.savefig(EDA_DIR / "fig11_pairplot.png"); plt.close(g.figure)


def fig12_course(df):
    """Dropout rate by course code (courses with >=100 students), with overall rate reference."""
    g = df.groupby("Course")[TARGET].agg(n="size", rate=lambda s: 100 * (s == "Dropout").mean())
    g = g[g.n >= 100].sort_values("rate")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(g.index.astype(str), g.rate, color="#eb6834", height=0.65)
    for y, (r, n) in enumerate(zip(g.rate, g.n)):
        ax.text(r + 0.6, y, f"{r:.0f}%  (n={n})", va="center", fontsize=8, bbox=dict(fc="white", ec="none", pad=0.8), zorder=5)
    overall = 100 * (df[TARGET] == "Dropout").mean()
    ax.axvline(overall, color="#0b0b0b", ls="--", lw=1.2); ax.text(overall + .4, -0.9, f"overall {overall:.1f}%", fontsize=8)
    ax.set(title="Fig. 12  Dropout rate by course code", xlabel="Dropout rate (%)", ylabel="Course code")
    ax.set_xlim(0, g.rate.max() * 1.25); ax.grid(axis="y", visible=False)
    save_fig(fig, EDA_DIR / "fig12_dropout_by_course.png")


def interactive(df):
    """Stage 7 (Interact): hover/zoom/filter pages built with Plotly."""
    import plotly.express as px
    d = df.copy(); d["Outcome"] = d[TARGET]
    f1 = px.scatter(d, x="Curricular units 1st sem (grade)", y="Curricular units 2nd sem (grade)",
                    color="Outcome", color_discrete_map=PAL, opacity=0.5,
                    hover_data=["Age at enrollment", "Admission grade", "Course", "Tuition fees up to date"],
                    title="Interactive: 1st vs 2nd semester grade (click legend to filter)")
    f1.write_html(EDA_DIR / "interactive_grades_scatter.html", include_plotlyjs="cdn")
    r = (d.groupby(["Course", "Outcome"]).size().reset_index(name="students"))
    f2 = px.bar(r, x="Course", y="students", color="Outcome", color_discrete_map=PAL,
                barmode="stack", title="Interactive: students by course code and outcome")
    f2.update_xaxes(type="category"); f2.write_html(EDA_DIR / "interactive_course_outcomes.html", include_plotlyjs="cdn")


def run():
    set_style()
    df, _ = clean_data(load_data())
    for f in (fig01_target, fig02_hist, fig03_box, fig04_outliers, fig05_normal, fig06_discrete,
              fig07_corr, fig08_binary, fig09_scatter, fig10_pca, fig11_pair, fig12_course, interactive):
        f(df); print("done", f.__name__)


if __name__ == "__main__":
    run()
