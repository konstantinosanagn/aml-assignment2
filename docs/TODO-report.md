# Assignment 2: Report Writing TODO

## Submission Checklist

- [ ] Write PDF report (strictly 5-7 pages)
- [ ] Embed all 8 figures from `figures/` into the PDF
- [ ] No code blocks in the PDF body
- [ ] Submit PDF on Gradescope
- [ ] Verify GitHub repo is public: https://github.com/konstantinosanagn/aml-assignment2

---

## Code Results Summary (reference for report writing)

### Dataset
- Bank Marketing (`bank-additional-full.csv`): 41,188 rows, 20 features + target
- Target: term deposit subscription (binary: yes/no)
- Class imbalance: 88.73% no / 11.27% yes (ratio 7.9:1)
- After preprocessing: 43 features, 28,830 train / 6,179 val / 6,179 test

### Best Model Configurations

**GBDT (XGBoost):**
- `learning_rate=0.1`, `n_estimators=100`, `max_depth=9`, `subsample=0.8`, `reg_alpha=0.1`, `reg_lambda=3`
- `scale_pos_weight=7.9`, `tree_method='hist'`, `early_stopping_rounds=20`

**MLP (sklearn):**
- `hidden_layer_sizes=(256, 128)`, `activation='tanh'`, `learning_rate_init=0.01`, `max_iter=1000`, `alpha=0.01`
- `solver='adam'`, `early_stopping=True`, `n_iter_no_change=20`
- Model stopped at iteration 27 (early stopping), never hit max_iter ceiling

### Final Metrics (test set)

| Metric | GBDT | MLP |
|---|---|---|
| Accuracy | 0.8951 | 0.9176 |
| Precision | 0.5211 | 0.6837 |
| Recall | **0.8534** | 0.5000 |
| F1-score | **0.6471** | 0.5776 |
| AUC-PR | **0.6766** | 0.6653 |
| Training Time | **12.88s** | 31.91s |

### Key Findings
- GBDT outperforms MLP on F1 (0.647 vs 0.578) and AUC-PR (0.677 vs 0.665)
- GBDT has much higher recall (0.853 vs 0.500) — catches more actual positives
- MLP has higher precision (0.684 vs 0.521) — fewer false positives
- Both models have high accuracy (~90%) but this is misleading due to class imbalance
- GBDT trains ~2.5x faster than MLP
- GBDT learning rate comparison: lr=0.01 converges slowly but smoothly, lr=0.3 converges fast but overshoots
- MLP depth/width: (128, 64) and (256, 128) perform similarly; (64,) underperforms
- MLP lr=0.1 causes divergence (matmul overflow warnings) — demonstrates L7's "too large = divergence"

---

## Report Sections

### 1. Introduction (~0.5 page)

**What to write:**
- Problem: binary classification — predict whether a bank customer subscribes to a term deposit
- Dataset: UCI Bank Marketing, 41K rows, 20 features (10 numeric + 10 categorical)
- Class imbalance: only 11.3% positive — mention this shapes metric choices
- Objective: compare GBDT and MLP on the same task

**No figures needed here.**

---

### 2. Methods (~1.5 pages)

**What to write:**

#### Preprocessing decisions (justify each one):

| Decision | Justification (reference lecture) |
|---|---|
| Dropped `default` column | 20.87% unknown, only 3 "yes" values — near-zero variance (L2: "drop if sparse") |
| Mode imputation for job, marital, education, housing, loan | Low missing rates (0.2%-4.2%), mode is appropriate for categorical features (L2) |
| Imputation fit on training set only | Prevents data leakage (L2: "imputation must be done on TRAINING DATA ONLY") |
| `pdays=999` → binary `was_previously_contacted` + pdays zeroed | 96.3% are 999 (sentinel value), raw pdays is degenerate. Binary feature captures the meaningful signal |
| Ordinal encoding for `education` | Natural ordering: illiterate < basic.4y < ... < university.degree (L2: preserves ordinal relationships) |
| One-hot encoding for job, marital, month, day_of_week, poutcome | Nominal features with no natural ordering (L2: "preserves all categories without assuming order") |
| Binary encoding for contact, housing, loan | Only 2 categories each — simple 0/1 is sufficient |
| `contact_rate = campaign / (previous + 1)` | Engineered feature capturing contact intensity (L2: "feature engineering often more impactful than algorithm choice") |
| 70/15/15 stratified split | Preserves 11.3% positive rate across all splits. Validation set for tuning, test set locked until final evaluation (L4) |
| StandardScaler for MLP, no scaling for GBDT | L7: NNs compute weighted sums — features on different scales cause gradient domination. L5: trees split on ordering, not magnitude. |

#### Why StandardScaler over MinMaxScaler:
- StandardScaler (z-score) is less sensitive to outliers than MinMaxScaler (L2)
- Adam optimizer works well with zero-mean data
- Numeric features like `nr.employed` and `euribor3m` have different ranges — standardization handles this robustly

**Embed figure:** None required, but could include a small table of unknown counts.

---

### 3. Results — GBDT (~1 page)

**What to write:**

#### Hyperparameter tuning:
- Used RandomizedSearchCV (30 iterations, 3-fold CV, F1 scoring)
- Best config: lr=0.1, 100 estimators, max_depth=9, subsample=0.8, reg_alpha=0.1, reg_lambda=3
- Best CV F1: 0.6318

#### Training dynamics:
- Describe the train vs val loss curve: both curves decrease, gap indicates variance (L4)
- Early stopping triggered — prevents overfitting (L5: pre-pruning concept)
- `scale_pos_weight=7.9` handles class imbalance by weighting minority class higher

#### Feature importance:
- Discuss which features dominate (expect: euribor3m, nr.employed, duration-related, poutcome)
- L5: "trees have built-in feature selection — naturally ignore features that don't reduce impurity"
- Note: importance can be biased toward high-cardinality features (L2)

#### Learning rate comparison:
- lr=0.01: slow convergence, needs many rounds, smooth curve
- lr=0.1: good balance of speed and stability
- lr=0.3: fastest convergence but risk of overshooting
- Connect to L5: "learning rate controls step size (shrinkage)"

**Embed figures:** `gbdt_train_val_loss.png`, `gbdt_feature_importance.png`, `gbdt_learning_rate_comparison.png`

---

### 4. Results — MLP (~1 page)

**What to write:**

#### Hyperparameter tuning:
- Used RandomizedSearchCV (30 iterations, 3-fold CV, F1 scoring)
- Best config: (256, 128), tanh, lr=0.01, max_iter=1000, alpha=0.01
- Best CV F1: 0.5819
- Model stopped at iteration 27 via early stopping — max_iter=1000 was ceiling, never reached

#### Training loss curve:
- Describe the training loss and validation score plots
- Loss decreases smoothly (Adam optimizer — L7: "fast AND smooth convergence")
- Validation score peaked early then plateaued, triggering early stopping
- Gap between train loss and val score indicates the model's generalization behavior

#### Depth/width comparison:
- (64,): underperforms — too few parameters to capture patterns (underfitting)
- (128, 64) and (256, 128, 64): similar performance — diminishing returns from depth
- Connect to L7: Universal Approximation Theorem — even one hidden layer can approximate any function, but depth helps learn hierarchical representations
- Connect to L4: this is the model complexity axis of the U-shaped bias-variance curve

#### Learning rate comparison:
- lr=0.001: slow convergence, smooth loss curve
- lr=0.01: good convergence, best balance
- lr=0.1: **divergence** — overflow warnings, loss explodes. Demonstrates L7: "too large → divergence"

#### Why tanh won over relu:
- Best model uses tanh — zero-centered outputs can speed convergence (L7)
- With alpha=0.01 (L2 regularization), the stronger regularization works better with tanh's bounded output range

**Embed figures:** `mlp_training_loss.png`, `mlp_depth_width_comparison.png`, `mlp_learning_rate_comparison.png`

---

### 5. GBDT vs MLP Comparison (~1.5 pages)

**What to write:**

#### Metrics table:
- Embed the comparison table (reproduce from notebook output)
- Highlight: GBDT wins on F1, AUC-PR, Recall; MLP wins on Precision, Accuracy
- Explain why accuracy (89-92%) is misleading: a model predicting all "no" gets 88.7% (L4)
- F1 and AUC-PR are the right metrics for imbalanced classification (L4)

#### Training time:
- GBDT: ~13s vs MLP: ~32s — GBDT is 2.5x faster
- Discuss why: tree-based models are computationally simpler on tabular data; MLP requires matrix multiplications through multiple layers with iterative gradient updates

#### Discussion point (a): When to prefer GBDT vs MLP
- GBDT excels on tabular data with mixed feature types — "tabular data is jagged" (L5)
- MLP excels on high-dimensional dense features (images, text embeddings) where spatial/temporal patterns exist — "NNs expect smoothness" (L5)
- For this Bank Marketing dataset (tabular, mixed types, moderate size), GBDT is the clear winner

#### Discussion point (b): Interpretability
- GBDT: built-in feature importance via impurity reduction (L5). Can directly see which features drive predictions. "Trees naturally ignore non-informative features"
- MLP: black box — stacked nonlinear transformations (L7). No direct feature importance. Would need external tools (SHAP, LIME) for interpretability

#### Discussion point (c): Categorical features and missing values
- GBDT (XGBoost): handles missing values natively — learns optimal split direction for NaN. Label-encoded categoricals work naturally with threshold splits
- MLP: cannot handle NaN — requires explicit imputation. Requires one-hot encoding for nominals, which can explode dimensionality (job alone adds 11 columns)
- Our pipeline imputed all NaNs, but in production GBDT's native handling is a significant advantage

#### Discussion point (d): Hyperparameter sensitivity
- MLP is more sensitive: lr=0.1 caused complete divergence; architecture choice matters (64 vs 128,64 vs 256,128,64 showed clear differences); requires careful scaling
- GBDT is more forgiving: reasonable defaults work well, early stopping prevents most overfitting, no scaling needed. Our GBDT achieved strong F1 even with default-adjacent parameters

**Embed figures:** `confusion_matrices.png`, `pr_curves.png`

---

### 6. Discussion (~0.5 page)

**What to write:**

#### Bias-variance reflection:
- L4 formula: Generalization Error = Bias² + Variance + Noise
- GBDT: boosting reduces bias through sequential residual correction (L5). Regularization (L1/L2, subsampling, early stopping) controls variance
- MLP: high capacity → low bias but high variance risk. Regularization (alpha=0.01) and early stopping (stopped at iter 27) were critical
- Train/val loss gaps in both models show the variance component

#### Limitations:
- Dataset is relatively small (41K rows) — MLP may benefit more from larger datasets
- Only explored sklearn's MLPClassifier — PyTorch/TensorFlow would allow more flexible architectures (batch norm, dropout, learning rate scheduling)
- Class imbalance handled via scale_pos_weight (GBDT) but no explicit handling for MLP (could try SMOTE or class_weight in future work)
- The `duration` feature was kept in final models — in a real deployment scenario, it should be excluded (not known before a call)

---

### 7. AI Tool Disclosure (~0.5 page)

**What to write:**
- **Tool used:** Claude Code (Anthropic) — CLI-based AI coding assistant
- **How it was used:**
  - Project scaffolding and repo structure
  - Code generation for data preparation, model training, evaluation, and visualization modules
  - Hyperparameter search configuration
  - Debugging and optimization (early stopping, hist tree method)
  - Cross-referencing implementation against assignment requirements and lecture content
- **Personal contributions:**
  - Dataset selection decision and analysis
  - Interpretation of all results and findings
  - All report writing and analysis narratives
  - Hyperparameter tuning rationale and bias-variance reasoning
  - Review and validation of all generated code

---

## Figures Checklist

All saved in `figures/` — embed each in the corresponding report section:

- [ ] `gbdt_train_val_loss.png` → Section 3 (GBDT Results)
- [ ] `gbdt_feature_importance.png` → Section 3 (GBDT Results)
- [ ] `gbdt_learning_rate_comparison.png` → Section 3 (GBDT Results)
- [ ] `mlp_training_loss.png` → Section 4 (MLP Results)
- [ ] `mlp_depth_width_comparison.png` → Section 4 (MLP Results)
- [ ] `mlp_learning_rate_comparison.png` → Section 4 (MLP Results)
- [ ] `confusion_matrices.png` → Section 5 (Comparison)
- [ ] `pr_curves.png` → Section 5 (Comparison)

---

## Grading Rubric Reminder

| Category | Weight | Covered by |
|---|---|---|
| Data Preparation | 15% | Section 2 (Methods) |
| GBDT Implementation | 15% | Section 3 (Results — GBDT) |
| MLP Implementation | 15% | Section 4 (Results — MLP) |
| GBDT vs MLP Comparison | 25% | Section 5 (Comparison) |
| Evaluation & Visualization | 20% | Figures embedded across Sections 3-5 |
| AI Tool Usage Disclosure | 10% | Section 7 (AI Disclosure) |

---

## Key Phrases to Use (from lectures)

- "Accuracy is misleading on imbalanced data" (L4)
- "Boosting reduces bias through sequential residual correction" (L5)
- "Trees excel at tabular data — tabular data is 'jagged'" (L5)
- "Neural networks compute weighted sums — features on different scales dominate gradients" (L7)
- "Trees split on ordering, not magnitude — scaling is unnecessary" (L5)
- "Adam = Momentum + RMSProp: fast convergence AND stability" (L7)
- "Generalization Error = Bias² + Variance + Noise" (L4)
- "Feature engineering is often more impactful than algorithm choice" (L2)
- "No data leakage: fit preprocessing on training set only" (L2)
- "Universal Approximation Theorem: a single hidden layer can approximate any function" (L7)
