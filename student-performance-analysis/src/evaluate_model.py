"""Evaluate the saved MLP on the held-out TEST set: metrics, 5 result figures, permutation importance."""
import json, os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             confusion_matrix, roc_curve, auc, roc_auc_score, classification_report)
from sklearn.preprocessing import label_binarize
from config import MODEL_DIR, EVAL_DIR, SEED, CLASS_NAMES, CLASS_COLORS, set_style, save_fig
from data_preprocessing import load_data, clean_data, split_data

CC = [CLASS_COLORS[c] for c in CLASS_NAMES]


def metrics(y, prob):
    pred = prob.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "macro_precision": p, "macro_recall": r,
            "macro_f1": f, "weighted_f1": f1_score(y, pred, average="weighted"),
            "roc_auc_ovr_macro": roc_auc_score(y, prob, multi_class="ovr", average="macro")}


def run():
    set_style()
    df, _ = clean_data(load_data())
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(df)
    pre = joblib.load(MODEL_DIR / "preprocessor.joblib")
    mlp = tf.keras.models.load_model(MODEL_DIR / "mlp_student_outcome.keras")
    hist = json.load(open(EVAL_DIR / "history.json"))
    summ = json.load(open(EVAL_DIR / "training_summary.json"))
    Xte, Xval = (pre.transform(a).astype("float32") for a in (X_te, X_val))
    yte, yval = y_te.values, y_val.values
    prob = mlp.predict(Xte, verbose=0); pred = prob.argmax(1)
    res = {"mlp_test": metrics(yte, prob), "mlp_val": metrics(yval, mlp.predict(Xval, verbose=0))}
    res["per_class_report"] = classification_report(yte, pred, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(yte, pred); res["confusion_matrix"] = cm.tolist()
    # baselines on the same test set
    comp = {"MLP (proposed)": res["mlp_test"]}
    for name, f in [("Logistic regression", "baseline_logistic.joblib"), ("Random forest", "baseline_random.joblib")]:
        m = joblib.load(MODEL_DIR / f); comp[name] = metrics(yte, m.predict_proba(pre.transform(X_te)))
    res["comparison_test"] = comp
    pd.DataFrame(comp).T.round(4).to_csv(EVAL_DIR / "table_model_comparison_test.csv")

    # ---- Fig 13/14: learning curves -----------------------------------------------------
    ep = np.arange(1, len(hist["loss"]) + 1); best_ep = int(np.argmin(hist["val_loss"])) + 1
    res["best_epoch"] = best_ep; res["epochs_run"] = len(ep)
    for key, title, yl, name in [("loss", "Fig. 13  Training vs validation loss", "Cross-entropy loss", "fig13_loss_vs_epoch.png"),
                                 ("accuracy", "Fig. 14  Training vs validation accuracy", "Accuracy", "fig14_accuracy_vs_epoch.png")]:
        fig, ax = plt.subplots(figsize=(6.2, 4))
        ax.plot(ep, hist[key], color="#2a78d6", lw=2, label="Training")
        ax.plot(ep, hist["val_" + key], color="#eb6834", lw=2, label="Validation")
        ax.axvline(best_ep, color="#52514e", ls=":", lw=1.2); ax.text(best_ep + .3, ax.get_ylim()[0] + .01, f"best epoch {best_ep}", fontsize=8)
        ax.set(title=title, xlabel="Epoch", ylabel=yl); ax.legend()
        save_fig(fig, EVAL_DIR / name)

    # ---- Fig 15: confusion matrix (counts + row-normalised) ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4))
    for ax, data, fmt, ttl in [(axes[0], cm, "d", "(a) Counts"), (axes[1], cm / cm.sum(1, keepdims=True), ".2f", "(b) Row-normalised (recall)")]:
        im = ax.imshow(data, cmap="Blues", vmin=0)
        for i in range(3):
            for j in range(3):
                v = data[i, j]; ax.text(j, i, format(v, fmt), ha="center", va="center",
                                        color="white" if v > data.max() * .55 else "#0b0b0b", fontsize=11)
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set(title=ttl, xlabel="Predicted", ylabel="Actual"); ax.grid(False)
    fig.suptitle("Fig. 15  Confusion matrix on the test set", fontweight="bold", y=1.03); fig.tight_layout()
    save_fig(fig, EVAL_DIR / "fig15_confusion_matrix.png")

    # ---- Fig 16: one-vs-rest ROC curves -------------------------------------------------
    Y = label_binarize(yte, classes=[0, 1, 2]); res["roc_auc_per_class"] = {}
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for i, c in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(Y[:, i], prob[:, i]); a = auc(fpr, tpr); res["roc_auc_per_class"][c] = a
        ax.plot(fpr, tpr, color=CC[i], lw=2, label=f"{c} (AUC = {a:.3f})")
    ax.plot([0, 1], [0, 1], color="#52514e", ls="--", lw=1, label="Chance")
    ax.set(title="Fig. 16  One-vs-rest ROC curves (test set)", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right"); save_fig(fig, EVAL_DIR / "fig16_roc_curves.png")

    # ---- Fig 17: confidence distribution of correct vs wrong predictions ----------------
    conf = prob.max(1); ok = pred == yte
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.hist(conf[ok], bins=20, alpha=.8, color="#2a78d6", label=f"Correct (n={ok.sum()})", edgecolor="white")
    ax.hist(conf[~ok], bins=20, alpha=.8, color="#eb6834", label=f"Wrong (n={(~ok).sum()})", edgecolor="white")
    ax.set(title="Fig. 17  Prediction confidence: correct vs wrong", xlabel="Max predicted probability", ylabel="Students")
    ax.legend(); save_fig(fig, EVAL_DIR / "fig17_confidence_distribution.png")
    res["mean_conf_correct"], res["mean_conf_wrong"] = float(conf[ok].mean()), float(conf[~ok].mean())

    # ---- Fig 18: permutation importance on RAW features (drop in macro-F1) -------------
    rng = np.random.default_rng(SEED); base = f1_score(yte, pred, average="macro"); imp = {}
    for col in X_te.columns:
        drops = []
        for _ in range(10):
            Xp = X_te.copy(); Xp[col] = rng.permutation(Xp[col].values)
            drops.append(base - f1_score(yte, mlp.predict(pre.transform(Xp).astype("float32"), verbose=0).argmax(1), average="macro"))
        imp[col] = (np.mean(drops), np.std(drops))
    imp = pd.DataFrame(imp, index=["mean_drop", "std"]).T.sort_values("mean_drop", ascending=False)
    imp.round(4).to_csv(EVAL_DIR / "table_permutation_importance.csv")
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(top.index, top.mean_drop, xerr=top["std"], color="#2a78d6", height=.65, error_kw={"lw": 1, "ecolor": "#52514e"})
    ax.set(title="Fig. 18  Permutation importance (top 12 features)", xlabel="Drop in test macro-F1 when feature is shuffled")
    ax.grid(axis="y", visible=False); save_fig(fig, EVAL_DIR / "fig18_permutation_importance.png")
    res["top_importance"] = imp.head(10).round(4).reset_index().rename(columns={"index": "feature"}).to_dict("records")

    # ---- Fig 19: tuning results ----------------------------------------------------------
    t = pd.read_csv(EVAL_DIR / "tuning_results.csv")
    t["cfg"] = t.hidden + " d=" + t.dropout.astype(str) + " lr=" + t.lr.astype(str)
    t = t.sort_values("val_macro_f1")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = ["#eb6834" if v == t.val_macro_f1.max() else "#2a78d6" for v in t.val_macro_f1]
    ax.barh(t.cfg, t.val_macro_f1, color=colors, height=.65)
    for y, v in enumerate(t.val_macro_f1): ax.text(v + .002, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set(title="Fig. 19  Hyperparameter search (validation macro-F1)", xlabel="Validation macro-F1",
           xlim=(t.val_macro_f1.min() - .03, t.val_macro_f1.max() + .03)); ax.grid(axis="y", visible=False)
    save_fig(fig, EVAL_DIR / "fig19_tuning_results.png")

    json.dump(res, open(EVAL_DIR / "results_model.json", "w"), indent=2, default=float)
    print(json.dumps({k: res[k] for k in ["mlp_test", "mlp_val", "comparison_test", "roc_auc_per_class", "best_epoch", "epochs_run"]}, indent=1, default=float))
    print(classification_report(yte, pred, target_names=CLASS_NAMES)); print(cm); print(imp.head(10))


if __name__ == "__main__":
    run()
