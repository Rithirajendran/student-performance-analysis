# Student Academic Performance Analysis and Dropout Prediction (EDA + Deep Learning)

**Student:** RITHI R · **College ID:** 24AD091 · **Department:** Artificial Intelligence and Data Science, KPR Institute of Engineering and Technology
**Course:** U21ADP05 – Exploratory Data Analysis and Visualization · **Assignment II – Case Study** · AY 2026–2027, Semester V

## What this project does
Explores the UCI dataset *Predict Students' Dropout and Academic Success* (4,424 students, 36 features, target = Dropout / Enrolled / Graduate),
visualises it following Ben Fry's seven stages (Acquire, Parse, Filter, Mine, Represent, Refine, Interact), analyses it statistically
(univariate, bivariate, multivariate, probability distributions) and trains a multilayer perceptron (MLP) to predict the outcome.

## Dataset
* Source: <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success> · DOI 10.24432/C5MC89 · licence CC BY 4.0
* File used: `data/data.csv` (4,424 rows × 37 columns, 0 missing values, 0 duplicates)
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
| MLP (proposed) | 0.756 | 0.719 | 0.723 | 0.715 | 0.895 |
| Logistic regression | 0.767 | 0.735 | 0.745 | 0.732 | 0.902 |
| Random forest | 0.803 | 0.753 | 0.745 | 0.748 | 0.907 |

Best MLP: hidden layers [128, 64, 32], dropout 0.2, learning rate 0.001, batch size 64, early stopping (best epoch 13).
The MLP is competitive with, but not better than, the simpler baselines on this small tabular dataset.

## Key insights
* 49.93% Graduate / 32.12% Dropout / 17.95% Enrolled – moderately imbalanced.
* Dropout rate: 86.55% when tuition fees are not up to date vs 24.74% when they are.
* Units approved in semesters 1 and 2 are the strongest signals (statistical tests and permutation importance agree).
* A semester's average grade is 0 exactly when no unit was approved – a meaningful zero, not missing data.
* Count variables are over-dispersed (not Poisson); admission grade is bell-shaped but right-skewed.

## Limitations
Single split/seed; the strongest features are known only after semester 1–2; one institution; code values are not interpreted; associations are not causal.

See `report/` for the full report.
