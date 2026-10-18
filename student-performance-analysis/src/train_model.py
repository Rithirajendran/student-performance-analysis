"""Train the MLP: leakage-safe preprocessing, grid-search hyperparameter tuning on the
VALIDATION set, baselines (logistic regression, random forest), save model + artefacts."""
import itertools, json, os, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from config import MODEL_DIR, EVAL_DIR, SEED, CLASS_NAMES
from data_preprocessing import load_data, clean_data, split_data, build_preprocessor

BATCH_SIZE, MAX_EPOCHS, PATIENCE = 64, 100, 10
GRID = {"hidden": [(64, 32), (128, 64), (128, 64, 32)],
        "dropout": [0.2, 0.4],
        "lr": [1e-3, 3e-4]}


def build_mlp(n_in, hidden, dropout, lr, n_classes=3):
    """MLP: [Dense(ReLU) -> BatchNorm -> Dropout] x L -> Dense(softmax)."""
    layers = [tf.keras.layers.Input((n_in,))]
    for h in hidden:
        layers += [tf.keras.layers.Dense(h, activation="relu",
                                         kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
                   tf.keras.layers.BatchNormalization(),
                   tf.keras.layers.Dropout(dropout)]
    layers.append(tf.keras.layers.Dense(n_classes, activation="softmax"))
    m = tf.keras.Sequential(layers)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def fit_one(Xtr, ytr, Xval, yval, cw, hidden, dropout, lr):
    tf.keras.utils.set_random_seed(SEED)
    m = build_mlp(Xtr.shape[1], hidden, dropout, lr)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                          restore_best_weights=True)
    h = m.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
              class_weight=cw, callbacks=[es], verbose=0)
    pred = m.predict(Xval, verbose=0).argmax(1)          # uses restored best-val-loss weights
    return m, h, f1_score(yval, pred, average="macro"), float((pred == yval).mean()), min(h.history["val_loss"])


def run():
    df, _ = clean_data(load_data())
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(df)
    pre = build_preprocessor(X_tr).fit(X_tr)              # fitted on TRAIN only (no leakage)
    Xtr, Xval, Xte = (pre.transform(a).astype("float32") for a in (X_tr, X_val, X_te))
    ytr, yval, yte = y_tr.values, y_val.values, y_te.values
    joblib.dump(pre, MODEL_DIR / "preprocessor.joblib")
    feat_names = list(pre.get_feature_names_out())
    sizes = {"train": len(ytr), "val": len(yval), "test": len(yte), "n_input_features": Xtr.shape[1],
             "n_raw_features": X_tr.shape[1]}
    print(sizes)
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(3), y=ytr)))

    # ---- grid search on validation set --------------------------------------------------
    rows, best = [], None
    for hidden, dr, lr in itertools.product(GRID["hidden"], GRID["dropout"], GRID["lr"]):
        t0 = time.time()
        m, h, f1, vacc, vl = fit_one(Xtr, ytr, Xval, yval, cw, hidden, dr, lr)
        rec = {"hidden": str(hidden), "dropout": dr, "lr": lr, "epochs_run": len(h.history["loss"]),
               "val_macro_f1": f1, "val_loss": vl, "val_acc": vacc,
               "seconds": round(time.time() - t0, 1)}
        rows.append(rec); print(rec)
        if best is None or (f1, -vl) > (best["f1"], -best["vl"]):
            best = {"f1": f1, "vl": vl, "model": m, "hist": h.history,
                    "cfg": {"hidden": list(hidden), "dropout": dr, "lr": lr}}
    pd.DataFrame(rows).to_csv(EVAL_DIR / "tuning_results.csv", index=False)

    # ---- baselines ---------------------------------------------------------------------
    base = {}
    for name, mdl in {"Logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                                  random_state=SEED),
                      "Random forest": RandomForestClassifier(300, class_weight="balanced",
                                                              random_state=SEED, n_jobs=-1)}.items():
        mdl.fit(Xtr, ytr); joblib.dump(mdl, MODEL_DIR / f"baseline_{name.split()[0].lower()}.joblib")
        base[name] = {"val_macro_f1": f1_score(yval, mdl.predict(Xval), average="macro")}

    best["model"].save(MODEL_DIR / "mlp_student_outcome.keras")
    json.dump({"best_config": best["cfg"], "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
               "early_stopping_patience": PATIENCE, "epochs_run": len(best["hist"]["loss"]),
               "val_macro_f1": best["f1"], "sizes": sizes, "baselines_val": base,
               "class_weights": {CLASS_NAMES[k]: float(v) for k, v in cw.items()},
               "n_params": int(best["model"].count_params())},
              open(EVAL_DIR / "training_summary.json", "w"), indent=2)
    json.dump({k: [float(x) for x in v] for k, v in best["hist"].items()},
              open(EVAL_DIR / "history.json", "w"))
    json.dump(feat_names, open(EVAL_DIR / "feature_names.json", "w"))
    print("best:", best["cfg"], "val macro-F1", round(best["f1"], 4))


if __name__ == "__main__":
    run()
