# Assignment 2: From Trees to Neural Networks — Design Spec

**Dataset:** Bank Marketing (`bank-additional-full.csv`) — 41,188 rows, 20 features, binary classification
**Target:** Predict whether a customer subscribes to a term deposit (`y`: yes/no)
**Class distribution:** 88.73% no / 11.27% yes (imbalance ratio 7.9:1)

---

## 1. Data Preparation (15%)

### 1.1 Load & Inspect
- Load with `pd.read_csv('bank-additional-full.csv', sep=';')`
- Print `.shape`, `.dtypes`, `.describe()`, `.info()`
- Identify that `"unknown"` strings are the missing value encoding (not NaN)

### 1.2 Handle Missing Values
**Lecture basis (L2):** "Imputation must be done on TRAINING DATA ONLY."

| Feature | Unknown % | Strategy | Justification |
|---|---|---|---|
| `default` | 20.87% | Drop column | Only 3 "yes" values in 41K — near-zero variance. L2: "Drop if sparse." |
| `education` | 4.20% | Mode imputation (fit on train) | L2: mode for categorical features |
| `housing` | 2.40% | Mode imputation (fit on train) | Same |
| `loan` | 2.40% | Mode imputation (fit on train) | Same |
| `job` | 0.80% | Mode imputation (fit on train) | Low rate, mode is safe |
| `marital` | 0.19% | Mode imputation (fit on train) | Trivial amount |

### 1.3 Handle Special Values
- **`pdays=999`** (96.3%): Create binary `was_previously_contacted`. For contacted clients keep actual pdays; for others set to 0.
- **`duration`**: Train models with and without. UCI docs warn it leaks target info. Discuss impact in report.

### 1.4 Encode Categorical Features
**Lecture basis (L2):** One-hot encoding "preserves all categories without assuming order."

| Feature | Cardinality | Encoding | Why |
|---|---|---|---|
| `education` | 8 | Ordinal (basic.4y=1 → university.degree=6) | Natural ordering exists |
| `job` | 12 | One-hot | Nominal, no ordering |
| `marital` | 4 | One-hot | Nominal |
| `contact` | 2 | Label encode (0/1) | Binary |
| `month` | 10 | One-hot | Nominal |
| `day_of_week` | 5 | One-hot | Nominal |
| `poutcome` | 3 | One-hot | Nominal |

### 1.5 Feature Engineering
**Lecture basis (L2):** "Often more impactful than algorithm choice."

- `contact_rate = campaign / (previous + 1)` — contact intensity ratio
- `was_previously_contacted` — from pdays transformation
- Correlation analysis on macro-economic features (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`)

### 1.6 Split Data
- 70 / 15 / 15 train / validation / test with `stratify=y`
- `random_state=42` for reproducibility
- **Lecture basis (L4):** "Test set is locked — only use once at the very end."

### 1.7 Feature Scaling (MLP only)
- Fit `StandardScaler` on training set only, then `.transform()` val and test
- **Lecture basis (L7):** "If features have vastly different scales, gradients will be dominated by large-scale features." Activation functions saturate with large inputs (sigmoid max derivative = 0.25).
- **Lecture basis (L5):** Trees don't need scaling — "they only care about relative ordering, not magnitude."
- Discuss this contrast explicitly in report (assignment requirement).

---

## 2. Gradient Boosted Decision Trees — GBDT (15%)

**Lecture basis (L5):** "Boosting = sequential learning. Each new tree fixes errors of previous ones. Final model = weighted sum of all trees."

### 2.1 Train XGBClassifier
- `xgboost.XGBClassifier` with `eval_metric='logloss'`
- `eval_set=[(X_val, y_val)]` for monitoring
- `early_stopping_rounds=20` — L5 pre-pruning: "stop growth before overfitting appears"
- `scale_pos_weight ≈ 7.9` for class imbalance
- `random_state=42`

### 2.2 Hyperparameter Exploration

| Parameter | Values | Lecture Rationale |
|---|---|---|
| `learning_rate` | 0.01, 0.1, 0.3 | L5: "shrinkage" — smaller = more conservative, less overfitting |
| `n_estimators` | 100, 300, 500, 1000 | L5: "more trees = lower bias, but risk of overfitting" |
| `max_depth` | 3, 5, 7, 9 | L5: "shallow = weak learners for better generalization" |
| `subsample` | 0.7, 0.8, 1.0 | L5: subsampling as regularization (borrows from bagging's variance reduction) |
| `reg_alpha` (L1) | 0, 0.1, 1.0 | L5: regularization to prevent overfitting. L3: L1 drives weights to zero (sparsity) |
| `reg_lambda` (L2) | 1, 3, 5 | L5: L2 regularization on leaf weights. L3: smooth shrinkage |

- Use `RandomizedSearchCV` or `GridSearchCV` (assignment hint)
- **Lecture basis (L4):** "Use cross-validation for fair comparison. Pick model with best average validation score."

### 2.3 Visualizations (3 required)

**a) Training vs. Validation Loss**
- From `model.evals_result()` — plot logloss per boosting round
- L4: gap between train/val = variance. Converging = good fit. Diverging = overfitting.

**b) Feature Importance**
- `xgb.plot_importance(model)` or `model.feature_importances_`
- L5: "Trees have built-in feature selection — naturally ignore features that don't reduce impurity."
- L2: "Importance scores can be biased toward high-cardinality features — interpret with care."

**c) Learning Rate Comparison**
- Train 3 models: lr=0.01, 0.1, 0.3 (fixed n_estimators)
- Overlay validation loss curves
- L5: lower lr = more conservative steps, needs more trees; higher = faster but risk of overshooting

### 2.4 Report Discussion
- Connect `max_depth` to bias-variance (L4): deeper = lower bias, higher variance → U-shaped curve
- L5: "Boosting reduces bias" (sequential residual correction) vs. bagging reduces variance
- Early stopping as pre-pruning (L5)
- L5: "Trees excel at tabular data — tabular data is 'jagged' and unstructured"

---

## 3. Multi-Layer Perceptron — MLP (15%)

**Lecture basis (L7):** "By stacking neurons, we can approximate any function (Universal Approximation Theorem)."

### 3.1 Train MLPClassifier
- `sklearn.neural_network.MLPClassifier`
- Use **scaled** features only
- L7: default solver is `adam` — "fast convergence AND stability"
- `random_state=42`, `max_iter=500`

### 3.2 Hyperparameter Exploration

| Parameter | Values | Lecture Rationale |
|---|---|---|
| `hidden_layer_sizes` | (64,), (128, 64), (256, 128, 64) | L7: depth = more abstract representations; width = capacity per layer |
| `activation` | 'relu', 'tanh' | L7: ReLU = "sparse, efficient, dominant." Tanh = "zero-centered, faster convergence" but vanishing gradients |
| `learning_rate_init` | 0.001, 0.01, 0.1 | L7: "Too small → slow convergence. Too large → divergence." |
| `max_iter` | 200, 500, 1000 | Controls training duration |
| `alpha` (L2) | 0.0001, 0.001, 0.01 | L3/L4: Ridge regularization to control variance |

- Use `RandomizedSearchCV` or `GridSearchCV`

### 3.3 Visualizations (2 required)

**a) Training Loss Curve**
- `model.loss_curve_` — plot cross-entropy loss vs. iteration
- L7: Adam should show "fast AND smooth convergence"
- Discuss: still decreasing at max_iter? → needs more iterations. Flat? → converged.

**b) Depth/Width vs. Validation Performance**
- Train (64,), (128, 64), (256, 128, 64) with same other params
- Plot validation F1 for each
- L7: "Deeper = more abstract features." L4: model complexity axis of U-shaped curve.

### 3.4 Report Discussion
- Scaling rationale: L7 weighted sums + activation saturation vs. L5 trees use ordering only
- ReLU vs tanh: L7 vanishing gradient tradeoffs
- Connect depth/width results to Universal Approximation Theorem (L7)

---

## 4. GBDT vs MLP Comparison (25%)

**Lecture basis (L5):** "Tabular → XGBoost. Image/Text → Neural Networks."

### 4.1 Side-by-Side Metrics Table
Same train/test split for both (L4: fair comparison).

| Metric | GBDT | MLP |
|---|---|---|
| Accuracy | — | — |
| Precision | — | — |
| Recall | — | — |
| F1-score | — | — |
| AUC-PR | — | — |

- L4: "Accuracy is misleading on imbalanced data"
- Use `classification_report` and `average_precision_score`

### 4.2 Training Time Comparison
- Time both with `time.time()`
- GBDT expected to be significantly faster on this tabular dataset

### 4.3 Discussion (4 required topics)

**a) When to prefer GBDT vs MLP:**
- L5: "Trees excel at tabular data — 'jagged.' NNs expect 'smoothness.'" GBDT wins on structured/mixed-type. MLP wins on high-dimensional dense features.

**b) Interpretability:**
- L5: GBDT has built-in feature importance (impurity reduction). "Naturally ignores non-informative features."
- L7: MLP is a black box — stacked nonlinear transformations.

**c) Categorical features and missing values:**
- L5: XGBoost handles NaN natively (learns optimal split direction). Trees handle label-encoded categoricals via threshold splits.
- L7: MLP requires explicit encoding and imputation. One-hot may explode dimensionality.

**d) Hyperparameter sensitivity:**
- L7: MLP is more sensitive — lr too large → divergence, architecture choice matters dramatically.
- L5: GBDT is more forgiving — reasonable defaults + early stopping prevent most issues.

---

## 5. Evaluation & Visualization (20%)

All visualizations embedded in PDF report (not just in code).

| # | Plot | Source |
|---|---|---|
| 1 | GBDT train vs. val loss curve | `evals_result()` |
| 2 | GBDT feature importance | `plot_importance()` |
| 3 | GBDT learning rate comparison (3 curves) | 3 models overlaid |
| 4 | MLP training loss curve | `loss_curve_` |
| 5 | MLP depth/width comparison | Bar chart of val F1 |
| 6 | Confusion matrices (side-by-side) | `confusion_matrix` |
| 7 | PR curves (both overlaid) | `precision_recall_curve` |

---

## 6. Report Structure (5-7 pages)

| Section | ~Pages | Content |
|---|---|---|
| Introduction | 0.5 | Problem definition, dataset description, class imbalance |
| Methods | 1.5 | Preprocessing with justifications, encoding, scaling rationale, feature engineering |
| Results — GBDT | 1 | Hyperparameter exploration, training dynamics, feature importance, best config |
| Results — MLP | 1 | Hyperparameter exploration, loss curves, depth/width analysis, best config |
| GBDT vs MLP Comparison | 1.5 | Metrics table, training time, 4 discussion topics |
| Discussion | 0.5 | Bias-variance reflection (L4: Bias^2 + Variance + Noise), limitations |
| AI Tool Disclosure | 0.5 | Tools used, personal contributions |

---

## 7. AI Tool Disclosure (10%)

Transparently list:
- Claude Code used for: plan creation, code scaffolding, debugging assistance
- Personal contributions: data analysis decisions, interpretation of results, report writing, hyperparameter tuning rationale

---

## Key Principles (from lectures, applied throughout)

- **No data leakage** (L2): fit all preprocessing on train only
- **Justify every decision** (L2): explain *why*, not just *what*
- **Bias-Variance awareness** (L4): Generalization Error = Bias^2 + Variance + Noise
- **Reproducibility**: `random_state=42` everywhere
- **Fair comparison** (L4): same splits, same preprocessing, test set used once at the end
