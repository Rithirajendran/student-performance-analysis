"""Write README.md from the saved result files (so every number in it is measured)."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1]
M = json.load(open(R / "outputs/model_evaluation/results_model.json")); T = json.load(open(R / "outputs/model_evaluation/training_summary.json"))
E = json.load(open(R / "outputs/eda/results_eda.json")); c = M["comparison_test"]; cfg = T["best_config"]
row = lambda n, v: f"| {n} | {v['accuracy']:.3f} | {v['macro_precision']:.3f} | {v['macro_recall']:.3f} | {v['macro_f1']:.3f} | {v['roc_auc_ovr_macro']:.3f} |"
txt = f"""# Student Academic Performance Analysis and Dropout Prediction (EDA + Deep Learning)

**Student:** RITHI R · **College ID:** 24AD091 · **Department:** Artificial Intelligence and Data Science, KPR Institute of Engineering and Technology
**Course:** U21ADP05 – Exploratory Data Analysis and Visualization · **Assignment II – Case Study** · AY 2026–2027, Semester V

## What this project does
Explores the UCI dataset *Predict Students' Dropout and Academic Success* (4,424 students, 36 features, target = Dropout / Enrolled / Graduate),
visualises it following Ben Fry's seven stages (Acquire, Parse, Filter, Mine, Represent, Refine, Interact), analyses it statistically
(univariate, bivariate, multivariate, probability distributions) and trains a multilayer perceptron (MLP) to predict the outcome.

## Dataset
* Source: <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success> · DOI 10.24432/C5MC89 · licence CC BY 4.0
* File used: `data/data.csv` ({E['shape'][0]:,} rows × {E['shape'][1]} columns, {E['clean_report']['missing_cells']} missing values, {E['clean_report']['duplicate_rows']} duplicates)
* Citation: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository.

## Project structure
```
data/                 dataset
src/                  config, data_preprocessing, eda, visualizations, train_model, evaluate_model, diagrams, run_all
outputs/eda/          EDA figures (1-12), tables, interactive HTML pages
outputs/model_evaluation/  evaluation figures (13-19), metrics
models/               trained MLP (.keras) and fitted preprocessor (.joblib)
report/               Word + PDF report and build scripts
```

## How to run
```bash
pip install -r requirements.txt
python src/run_all.py        # EDA -> figures -> tuning/training -> evaluation (a few minutes on CPU)
```

## Results (held-out test set, 664 students, seed 42, single stratified 70/15/15 split)
| Model | Accuracy | Macro precision | Macro recall | Macro F1 | ROC-AUC (OvR) |
|---|---|---|---|---|---|
{row('MLP (proposed)', c['MLP (proposed)'])}
{row('Logistic regression', c['Logistic regression'])}
{row('Random forest', c['Random forest'])}

Best MLP: hidden layers {cfg['hidden']}, dropout {cfg['dropout']}, learning rate {cfg['lr']}, batch size {T['batch_size']}, early stopping (best epoch {M['best_epoch']}).
The MLP is competitive with, but not better than, the simpler baselines on this small tabular dataset.

## Key insights
* {E['class_pct']['Graduate']}% Graduate / {E['class_pct']['Dropout']}% Dropout / {E['class_pct']['Enrolled']}% Enrolled – moderately imbalanced.
* Dropout rate: {E['dropout_rate_by_binary']['Tuition fees up to date']['0']}% when tuition fees are not up to date vs {E['dropout_rate_by_binary']['Tuition fees up to date']['1']}% when they are.
* Units approved in semesters 1 and 2 are the strongest signals (statistical tests and permutation importance agree).
* A semester's average grade is 0 exactly when no unit was approved – a meaningful zero, not missing data.
* Count variables are over-dispersed (not Poisson); admission grade is bell-shaped but right-skewed.

## Limitations
Single split/seed; the strongest features are known only after semester 1–2; one institution; code values are not interpreted; associations are not causal.

See `report/` for the full report.
"""
(R / "README.md").write_text(txt); print("README written")
