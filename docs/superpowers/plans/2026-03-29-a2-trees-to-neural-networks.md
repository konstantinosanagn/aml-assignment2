# Assignment 2: From Trees to Neural Networks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete GBDT vs MLP comparison pipeline on the Bank Marketing dataset, producing a runnable notebook and all visualizations for the 5-7 page PDF report.

**Architecture:** Modular `src/` package with one file per rubric section (data_preparation, gbdt_model, mlp_model, evaluation), orchestrated by a single Jupyter notebook. Figures saved to `figures/` for report embedding.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, jupyter

---

## File Structure

| File | Responsibility |
|---|---|
| `src/__init__.py` | Package marker |
| `src/utils.py` | Random seed, plot styling, timing decorator, shared constants |
| `src/data_preparation.py` | Load CSV, clean, encode, engineer features, split, scale |
| `src/gbdt_model.py` | XGBoost training, hyperparameter search, GBDT-specific plots |
| `src/mlp_model.py` | MLP training, hyperparameter search, MLP-specific plots |
| `src/evaluation.py` | Metrics computation, comparison table, PR curves, confusion matrices |
| `notebook.ipynb` | Orchestration — imports src, runs pipeline, displays all results |
| `requirements.txt` | Pinned dependencies |
| `.gitignore` | Ignore data/, figures/, checkpoints, __pycache__ |
| `README.md` | Project description, setup instructions, dataset download |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/__init__.py`
- Create: `src/utils.py`

- [ ] **Step 1: Create requirements.txt**

```txt
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
```

- [ ] **Step 2: Create .gitignore**

```
data/
figures/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/
dist/
build/
.DS_Store
```

- [ ] **Step 3: Create README.md**

```markdown
# Assignment 2: From Trees to Neural Networks

COMS 4995 Applied Machine Learning — Spring 2026

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download `bank-additional-full.csv` from:
https://archive.ics.uci.edu/dataset/222/bank+marketing

Place it in `data/bank-additional-full.csv`.

## Run

Open `notebook.ipynb` and run all cells.
```

- [ ] **Step 4: Create src/__init__.py**

```python
"""Assignment 2: From Trees to Neural Networks — source package."""
```

- [ ] **Step 5: Create src/utils.py**

```python
"""Shared utilities: random seed, plot styling, timing."""

import time
import functools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
FIGURES_DIR = "figures"

def set_global_seed(seed=RANDOM_STATE):
    """Set random seed for reproducibility."""
    np.random.seed(seed)

def setup_plotting():
    """Configure matplotlib/seaborn for consistent, clean plots."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

def save_figure(fig, filename):
    """Save figure to figures/ directory."""
    import os
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path)
    print(f"Saved: {path}")
    return path

def timer(func):
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} completed in {elapsed:.2f}s")
        return result, elapsed
    return wrapper
```

- [ ] **Step 6: Create data/ and figures/ directories**

Run: `mkdir -p data figures`

- [ ] **Step 7: Download the dataset**

Run: `curl -L -o data/bank-additional.zip https://archive.ics.uci.edu/static/public/222/bank+marketing.zip && cd data && unzip -o bank-additional.zip && find . -name 'bank-additional-full.csv' -exec mv {} bank-additional-full.csv \;`

Verify: `head -1 data/bank-additional-full.csv` should show semicolon-separated headers.

- [ ] **Step 8: Install dependencies**

Run: `pip install -r requirements.txt`

- [ ] **Step 9: Commit**

```bash
git add requirements.txt .gitignore README.md src/__init__.py src/utils.py
git commit -m "scaffold: project structure, dependencies, and shared utils"
```

---

### Task 2: Data Preparation Module

**Files:**
- Create: `src/data_preparation.py`

- [ ] **Step 1: Create src/data_preparation.py**

```python
"""Data loading, cleaning, encoding, feature engineering, splitting, and scaling."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from src.utils import RANDOM_STATE


def load_data(filepath="data/bank-additional-full.csv"):
    """Load the Bank Marketing dataset."""
    df = pd.read_csv(filepath, sep=";")
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_data(df):
    """Print dataset overview and missing value summary."""
    print("\n--- Data Types ---")
    print(df.dtypes)
    print("\n--- Descriptive Statistics ---")
    print(df.describe(include="all"))
    print("\n--- 'unknown' counts per column ---")
    unknown_counts = {}
    for col in df.select_dtypes(include="object").columns:
        count = (df[col] == "unknown").sum()
        if count > 0:
            unknown_counts[col] = (count, f"{count / len(df) * 100:.2f}%")
    for col, (count, pct) in sorted(unknown_counts.items(), key=lambda x: -x[1][0]):
        print(f"  {col}: {count} ({pct})")
    return unknown_counts


def clean_data(df):
    """Handle missing values and special encodings.

    - Drop 'default' column (20.87% unknown, only 3 'yes' values — near-zero variance).
    - Replace 'unknown' with NaN for later imputation.
    - Transform pdays: create 'was_previously_contacted' binary feature,
      set pdays=0 for non-contacted clients.
    """
    df = df.copy()

    # Drop 'default' — near-zero variance (only 3 'yes' in 41K rows)
    df = df.drop(columns=["default"])

    # Replace 'unknown' strings with NaN for imputation
    df = df.replace("unknown", np.nan)

    # Transform pdays: 999 = never contacted
    df["was_previously_contacted"] = (df["pdays"] != 999).astype(int)
    df.loc[df["pdays"] == 999, "pdays"] = 0

    # Encode target: 'yes' -> 1, 'no' -> 0
    df["y"] = (df["y"] == "yes").astype(int)

    return df


def impute_missing(df_train, df_val, df_test):
    """Impute categorical NaNs with mode from training set only (no data leakage).

    Features imputed: job, marital, education, housing, loan.
    """
    cat_cols_to_impute = ["job", "marital", "education", "housing", "loan"]
    modes = {}
    for col in cat_cols_to_impute:
        if col in df_train.columns:
            modes[col] = df_train[col].mode()[0]

    for df in [df_train, df_val, df_test]:
        for col, mode_val in modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode_val)

    print(f"Imputed columns with training modes: {modes}")
    return df_train, df_val, df_test


def encode_features(df_train, df_val, df_test):
    """Encode categorical features.

    - education: ordinal (natural ordering exists)
    - contact: label encode (binary)
    - job, marital, month, day_of_week, poutcome: one-hot
    """
    # Ordinal encoding for education
    edu_order = [
        "illiterate", "basic.4y", "basic.6y", "basic.9y",
        "high.school", "professional.course", "university.degree"
    ]
    for df in [df_train, df_val, df_test]:
        df["education"] = df["education"].map(
            {level: i for i, level in enumerate(edu_order)}
        )

    # Label encode contact (binary: cellular=1, telephone=0)
    for df in [df_train, df_val, df_test]:
        df["contact"] = (df["contact"] == "cellular").astype(int)

    # One-hot encode nominal categoricals
    ohe_cols = ["job", "marital", "month", "day_of_week", "poutcome"]
    df_train = pd.get_dummies(df_train, columns=ohe_cols, drop_first=True, dtype=int)
    df_val = pd.get_dummies(df_val, columns=ohe_cols, drop_first=True, dtype=int)
    df_test = pd.get_dummies(df_test, columns=ohe_cols, drop_first=True, dtype=int)

    # Align columns across splits (handle unseen categories)
    for col in df_train.columns:
        if col not in df_val.columns:
            df_val[col] = 0
        if col not in df_test.columns:
            df_test[col] = 0
    df_val = df_val[df_train.columns]
    df_test = df_test[df_train.columns]

    print(f"Encoded features. Final shape: {df_train.shape[1]} columns")
    return df_train, df_val, df_test


def engineer_features(df):
    """Create new features before splitting.

    - contact_rate: campaign / (previous + 1)
    """
    df = df.copy()
    df["contact_rate"] = df["campaign"] / (df["previous"] + 1)
    return df


def split_data(df, target_col="y", train_size=0.70, val_size=0.15, test_size=0.15):
    """Split into train/val/test with stratification.

    70/15/15 split. Stratified to preserve class ratios.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    # Second split: train vs val (val_size relative to remaining)
    val_relative = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"Split sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Positive rate — Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Standardize features for MLP. Fit on train only (no data leakage).

    Returns scaled copies and the fitted scaler.
    """
    scaler = StandardScaler()
    feature_cols = X_train.columns

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=feature_cols, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    print("Features standardized (fit on train only)")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def prepare_data(filepath="data/bank-additional-full.csv"):
    """Full data preparation pipeline. Returns all splits for both GBDT and MLP.

    Returns:
        dict with keys:
            'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test' — for GBDT
            'X_train_scaled', 'X_val_scaled', 'X_test_scaled' — for MLP
            'scaler' — fitted StandardScaler
            'feature_names' — list of feature column names
    """
    # Load and inspect
    df = load_data(filepath)
    inspect_data(df)

    # Clean
    df = clean_data(df)

    # Feature engineering (before split to avoid column mismatch)
    df = engineer_features(df)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Impute (fit on train only)
    X_train, X_val, X_test = impute_missing(X_train, X_val, X_test)

    # Encode
    X_train, X_val, X_test = encode_features(X_train, X_val, X_test)

    # Scale for MLP
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "feature_names": list(X_train.columns),
    }
```

- [ ] **Step 2: Verify data preparation runs**

Run: `python -c "from src.data_preparation import prepare_data; data = prepare_data(); print('Features:', len(data['feature_names']), '| Train:', data['X_train'].shape)"`

Expected: Prints dataset stats, split sizes, positive rates, and final feature count (~50 columns after one-hot encoding).

- [ ] **Step 3: Commit**

```bash
git add src/data_preparation.py
git commit -m "feat: add data preparation module with cleaning, encoding, and scaling"
```

---

### Task 3: GBDT Model Module

**Files:**
- Create: `src/gbdt_model.py`

- [ ] **Step 1: Create src/gbdt_model.py**

```python
"""XGBoost GBDT training, hyperparameter tuning, and visualizations."""

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from src.utils import RANDOM_STATE, save_figure, setup_plotting


def train_gbdt(X_train, y_train, X_val, y_val, params=None):
    """Train an XGBClassifier with eval_set monitoring and early stopping.

    Default params tuned for the Bank Marketing dataset (imbalanced, tabular).
    """
    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 3,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "use_label_encoder": False,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    return model


def tune_gbdt(X_train, y_train, X_val, y_val):
    """Hyperparameter search using RandomizedSearchCV.

    Explores: learning_rate, n_estimators, max_depth, subsample, reg_alpha, reg_lambda.
    """
    param_distributions = {
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "n_estimators": [100, 300, 500, 1000],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1, 3, 5],
    }

    base_model = xgb.XGBClassifier(
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        use_label_encoder=False,
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=30,
        scoring="f1",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV F1: {search.best_score_:.4f}")

    # Retrain best model with eval_set for loss curves
    best_model = train_gbdt(X_train, y_train, X_val, y_val, params=search.best_params_)
    return best_model, search


def plot_training_validation_loss(model):
    """Plot training vs. validation logloss per boosting round.

    Shows convergence behavior and overfitting detection.
    Gap between curves = variance (L4).
    """
    setup_plotting()
    results = model.evals_result()
    train_loss = results["validation_0"]["logloss"]
    val_loss = results["validation_1"]["logloss"]

    fig, ax = plt.subplots()
    ax.plot(train_loss, label="Train Loss", linewidth=2)
    ax.plot(val_loss, label="Validation Loss", linewidth=2)
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("Log Loss")
    ax.set_title("GBDT: Training vs. Validation Loss")
    ax.legend()
    save_figure(fig, "gbdt_train_val_loss.png")
    plt.show()
    return fig


def plot_feature_importance(model, max_features=15):
    """Plot top feature importances from the trained GBDT model.

    Trees naturally rank features by impurity reduction (L5).
    """
    setup_plotting()
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(
        model, ax=ax, max_num_features=max_features,
        importance_type="gain", title="GBDT: Feature Importance (Gain)"
    )
    save_figure(fig, "gbdt_feature_importance.png")
    plt.show()
    return fig


def plot_learning_rate_comparison(X_train, y_train, X_val, y_val):
    """Compare validation loss curves for learning_rate = 0.01, 0.1, 0.3.

    Lower lr = conservative steps, needs more trees (L5: shrinkage).
    Higher lr = faster convergence, risk of overshooting.
    """
    setup_plotting()
    learning_rates = [0.01, 0.1, 0.3]
    fig, ax = plt.subplots()

    for lr in learning_rates:
        model = train_gbdt(
            X_train, y_train, X_val, y_val,
            params={"learning_rate": lr, "n_estimators": 500}
        )
        val_loss = model.evals_result()["validation_1"]["logloss"]
        ax.plot(val_loss, label=f"lr={lr}", linewidth=2)

    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("Validation Log Loss")
    ax.set_title("GBDT: Effect of Learning Rate on Convergence")
    ax.legend()
    save_figure(fig, "gbdt_learning_rate_comparison.png")
    plt.show()
    return fig
```

- [ ] **Step 2: Verify GBDT module loads**

Run: `python -c "from src.gbdt_model import train_gbdt; print('GBDT module loaded successfully')"`

Expected: `GBDT module loaded successfully`

- [ ] **Step 3: Commit**

```bash
git add src/gbdt_model.py
git commit -m "feat: add GBDT module with XGBoost training, tuning, and visualizations"
```

---

### Task 4: MLP Model Module

**Files:**
- Create: `src/mlp_model.py`

- [ ] **Step 1: Create src/mlp_model.py**

```python
"""Scikit-learn MLP training, hyperparameter tuning, and visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from src.utils import RANDOM_STATE, save_figure, setup_plotting


def train_mlp(X_train, y_train, params=None):
    """Train an MLPClassifier with Adam optimizer.

    Input features MUST be standardized (L7: weighted sums sensitive to scale).
    """
    default_params = {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "solver": "adam",
        "learning_rate_init": 0.001,
        "max_iter": 500,
        "random_state": RANDOM_STATE,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20,
    }
    if params:
        default_params.update(params)

    model = MLPClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def tune_mlp(X_train, y_train):
    """Hyperparameter search using RandomizedSearchCV.

    Explores: hidden_layer_sizes, activation, learning_rate_init, alpha.
    """
    param_distributions = {
        "hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64), (128,), (256, 128)],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "alpha": [0.0001, 0.001, 0.01],
    }

    base_model = MLPClassifier(
        solver="adam",
        max_iter=500,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=20,
        scoring="f1",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV F1: {search.best_score_:.4f}")

    # Retrain best model to get loss_curve_
    best_model = train_mlp(X_train, y_train, params=search.best_params_)
    return best_model, search


def plot_training_loss_curve(model):
    """Plot MLP training loss curve from loss_curve_ attribute.

    Adam optimizer should show fast AND smooth convergence (L7).
    """
    setup_plotting()
    fig, ax = plt.subplots()
    ax.plot(model.loss_curve_, linewidth=2, color="coral")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("MLP: Training Loss Curve")
    save_figure(fig, "mlp_training_loss.png")
    plt.show()
    return fig


def plot_depth_width_comparison(X_train, y_train, X_val, y_val):
    """Compare validation F1 across different network architectures.

    Deeper networks learn more abstract representations (L7),
    but risk overfitting (L4: U-shaped complexity curve).
    """
    setup_plotting()
    architectures = {
        "(64,)": (64,),
        "(128, 64)": (128, 64),
        "(256, 128, 64)": (256, 128, 64),
    }

    from sklearn.metrics import f1_score
    results = {}
    loss_curves = {}

    for name, layers in architectures.items():
        model = train_mlp(X_train, y_train, params={"hidden_layer_sizes": layers})
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        results[name] = f1
        loss_curves[name] = model.loss_curve_
        print(f"  {name}: Val F1 = {f1:.4f}")

    # Bar chart of F1 scores
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(results.keys(), results.values(), color=["steelblue", "coral", "seagreen"])
    axes[0].set_xlabel("Architecture")
    axes[0].set_ylabel("Validation F1 Score")
    axes[0].set_title("MLP: Effect of Depth/Width on Validation F1")
    axes[0].set_ylim(0, max(results.values()) * 1.2)
    for i, (name, val) in enumerate(results.items()):
        axes[0].text(i, val + 0.005, f"{val:.4f}", ha="center", fontsize=10)

    # Loss curves overlay
    for name, curve in loss_curves.items():
        axes[1].plot(curve, label=name, linewidth=2)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].set_title("MLP: Training Loss by Architecture")
    axes[1].legend()

    plt.tight_layout()
    save_figure(fig, "mlp_depth_width_comparison.png")
    plt.show()
    return fig, results


def plot_learning_rate_comparison(X_train, y_train, X_val, y_val):
    """Compare MLP training for learning_rate_init = 0.001, 0.01, 0.1.

    Too small = slow convergence. Too large = divergence (L7).
    """
    setup_plotting()
    learning_rates = [0.001, 0.01, 0.1]
    fig, ax = plt.subplots()

    from sklearn.metrics import f1_score
    for lr in learning_rates:
        model = train_mlp(X_train, y_train, params={"learning_rate_init": lr})
        ax.plot(model.loss_curve_, label=f"lr={lr}", linewidth=2)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        print(f"  lr={lr}: Val F1 = {f1:.4f}, iterations = {model.n_iter_}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("MLP: Effect of Learning Rate on Training")
    ax.legend()
    save_figure(fig, "mlp_learning_rate_comparison.png")
    plt.show()
    return fig
```

- [ ] **Step 2: Verify MLP module loads**

Run: `python -c "from src.mlp_model import train_mlp; print('MLP module loaded successfully')"`

Expected: `MLP module loaded successfully`

- [ ] **Step 3: Commit**

```bash
git add src/mlp_model.py
git commit -m "feat: add MLP module with sklearn MLPClassifier training, tuning, and visualizations"
```

---

### Task 5: Evaluation Module

**Files:**
- Create: `src/evaluation.py`

- [ ] **Step 1: Create src/evaluation.py**

```python
"""Model evaluation, comparison table, PR curves, and confusion matrices."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report,
)
from src.utils import save_figure, setup_plotting


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all required classification metrics.

    Returns dict with: accuracy, precision, recall, f1, auc_pr.
    L4: Use multiple metrics — accuracy alone is misleading on imbalanced data.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "AUC-PR": average_precision_score(y_true, y_prob),
    }


def build_comparison_table(y_test, gbdt_pred, gbdt_prob, mlp_pred, mlp_prob,
                           gbdt_time, mlp_time):
    """Build side-by-side comparison table for GBDT vs MLP.

    Same test set, same preprocessing — fair comparison (L4).
    """
    gbdt_metrics = compute_metrics(y_test, gbdt_pred, gbdt_prob)
    mlp_metrics = compute_metrics(y_test, mlp_pred, mlp_prob)

    gbdt_metrics["Training Time (s)"] = f"{gbdt_time:.2f}"
    mlp_metrics["Training Time (s)"] = f"{mlp_time:.2f}"

    table = pd.DataFrame({"GBDT (XGBoost)": gbdt_metrics, "MLP (sklearn)": mlp_metrics})

    # Format numeric values
    for col in table.columns:
        for idx in table.index:
            val = table.loc[idx, col]
            if isinstance(val, float):
                table.loc[idx, col] = f"{val:.4f}"

    print("\n=== GBDT vs MLP Comparison ===")
    print(table.to_string())
    return table


def plot_confusion_matrices(y_test, gbdt_pred, mlp_pred):
    """Plot side-by-side confusion matrices for both models."""
    setup_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, preds, title in [
        (axes[0], gbdt_pred, "GBDT (XGBoost)"),
        (axes[1], mlp_pred, "MLP (sklearn)"),
    ]:
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No (0)", "Yes (1)"],
            yticklabels=["No (0)", "Yes (1)"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix: {title}")

    plt.tight_layout()
    save_figure(fig, "confusion_matrices.png")
    plt.show()
    return fig


def plot_pr_curves(y_test, gbdt_prob, mlp_prob):
    """Plot overlaid Precision-Recall curves for both models.

    AUC-PR is the right metric for imbalanced classification (L4).
    """
    setup_plotting()
    fig, ax = plt.subplots()

    for probs, label, color in [
        (gbdt_prob, "GBDT", "steelblue"),
        (mlp_prob, "MLP", "coral"),
    ]:
        precision, recall, _ = precision_recall_curve(y_test, probs)
        auc_pr = average_precision_score(y_test, probs)
        ax.plot(recall, precision, label=f"{label} (AUC-PR={auc_pr:.3f})",
                linewidth=2, color=color)

    # Baseline: proportion of positives
    baseline = y_test.mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves: GBDT vs MLP")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_figure(fig, "pr_curves.png")
    plt.show()
    return fig


def print_classification_reports(y_test, gbdt_pred, mlp_pred):
    """Print full classification reports for both models."""
    print("\n=== GBDT Classification Report ===")
    print(classification_report(y_test, gbdt_pred, target_names=["No", "Yes"]))
    print("\n=== MLP Classification Report ===")
    print(classification_report(y_test, mlp_pred, target_names=["No", "Yes"]))
```

- [ ] **Step 2: Verify evaluation module loads**

Run: `python -c "from src.evaluation import compute_metrics; print('Evaluation module loaded successfully')"`

Expected: `Evaluation module loaded successfully`

- [ ] **Step 3: Commit**

```bash
git add src/evaluation.py
git commit -m "feat: add evaluation module with metrics, comparison table, PR curves, confusion matrices"
```

---

### Task 6: Orchestration Notebook

**Files:**
- Create: `notebook.ipynb`

- [ ] **Step 1: Create notebook.ipynb**

The notebook has 7 sections corresponding to the rubric. Each cell imports from `src/` and runs the pipeline.

**Cell 1 — Setup & Imports:**

```python
# Assignment 2: From Trees to Neural Networks
# COMS 4995 Applied Machine Learning — Spring 2026

import warnings
warnings.filterwarnings("ignore")

from src.utils import set_global_seed, setup_plotting, timer
from src.data_preparation import prepare_data, load_data, inspect_data, clean_data
from src.gbdt_model import train_gbdt, tune_gbdt, plot_training_validation_loss, plot_feature_importance, plot_learning_rate_comparison
from src.mlp_model import train_mlp, tune_mlp, plot_training_loss_curve, plot_depth_width_comparison
from src.mlp_model import plot_learning_rate_comparison as mlp_plot_lr
from src.evaluation import (
    build_comparison_table, plot_confusion_matrices, plot_pr_curves,
    print_classification_reports,
)

set_global_seed()
setup_plotting()
print("All modules loaded successfully.")
```

**Cell 2 — Data Preparation (Section 1):**

```python
# === 1. DATA PREPARATION ===
data = prepare_data()

X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
X_train_scaled = data["X_train_scaled"]
X_val_scaled = data["X_val_scaled"]
X_test_scaled = data["X_test_scaled"]

print(f"\nFeature count: {len(data['feature_names'])}")
print(f"Class balance — Positive: {y_train.mean():.3f}, Negative: {1 - y_train.mean():.3f}")
```

**Cell 3 — GBDT Training & Tuning (Section 2):**

```python
# === 2. GRADIENT BOOSTED DECISION TREES ===

import time

# Hyperparameter tuning
print("Tuning GBDT hyperparameters...")
start = time.time()
gbdt_model, gbdt_search = tune_gbdt(X_train, y_train, X_val, y_val)
gbdt_time = time.time() - start
print(f"\nGBDT total tuning + training time: {gbdt_time:.2f}s")
```

**Cell 4 — GBDT Visualizations:**

```python
# GBDT Visualizations
print("--- Training vs. Validation Loss ---")
plot_training_validation_loss(gbdt_model)

print("\n--- Feature Importance ---")
plot_feature_importance(gbdt_model)

print("\n--- Learning Rate Comparison ---")
plot_learning_rate_comparison(X_train, y_train, X_val, y_val)
```

**Cell 5 — MLP Training & Tuning (Section 3):**

```python
# === 3. MULTI-LAYER PERCEPTRON ===

print("Tuning MLP hyperparameters...")
start = time.time()
mlp_model, mlp_search = tune_mlp(X_train_scaled, y_train)
mlp_time = time.time() - start
print(f"\nMLP total tuning + training time: {mlp_time:.2f}s")
```

**Cell 6 — MLP Visualizations:**

```python
# MLP Visualizations
print("--- Training Loss Curve ---")
plot_training_loss_curve(mlp_model)

print("\n--- Depth/Width Comparison ---")
plot_depth_width_comparison(X_train_scaled, y_train, X_val_scaled, y_val)

print("\n--- Learning Rate Comparison ---")
mlp_plot_lr(X_train_scaled, y_train, X_val_scaled, y_val)
```

**Cell 7 — GBDT vs MLP Comparison (Section 4):**

```python
# === 4. GBDT vs MLP COMPARISON ===

# Predictions
gbdt_pred = gbdt_model.predict(X_test)
gbdt_prob = gbdt_model.predict_proba(X_test)[:, 1]

mlp_pred = mlp_model.predict(X_test_scaled)
mlp_prob = mlp_model.predict_proba(X_test_scaled)[:, 1]

# Comparison table
comparison = build_comparison_table(
    y_test, gbdt_pred, gbdt_prob, mlp_pred, mlp_prob, gbdt_time, mlp_time
)
```

**Cell 8 — Evaluation Visualizations (Section 5):**

```python
# === 5. EVALUATION & VISUALIZATION ===

print("--- Classification Reports ---")
print_classification_reports(y_test, gbdt_pred, mlp_pred)

print("\n--- Confusion Matrices ---")
plot_confusion_matrices(y_test, gbdt_pred, mlp_pred)

print("\n--- Precision-Recall Curves ---")
plot_pr_curves(y_test, gbdt_prob, mlp_prob)
```

**Cell 9 — Summary:**

```python
# === SUMMARY ===
print("\n" + "="*60)
print("ASSIGNMENT 2 COMPLETE — All visualizations saved to figures/")
print("="*60)
print(f"\nBest GBDT params: {gbdt_search.best_params_}")
print(f"Best MLP params:  {mlp_search.best_params_}")
print(f"\nGBDT Test F1: {comparison.loc['F1-score', 'GBDT (XGBoost)']}")
print(f"MLP  Test F1: {comparison.loc['F1-score', 'MLP (sklearn)']}")
```

- [ ] **Step 2: Verify notebook runs end-to-end**

Run: `jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook_executed.ipynb`

Expected: Notebook executes without errors. All 7 figures saved to `figures/`. Comparison table printed.

- [ ] **Step 3: Commit**

```bash
git add notebook.ipynb
git commit -m "feat: add orchestration notebook running full GBDT vs MLP pipeline"
```

---

### Task 7: End-to-End Verification & Cleanup

**Files:**
- Modify: `notebook.ipynb` (if needed)
- Verify: `figures/` contains all 7 plots

- [ ] **Step 1: Run full notebook and verify outputs**

Run: `jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook_executed.ipynb 2>&1 | tail -5`

Expected: No errors. Check that all figures were saved:

Run: `ls -la figures/`

Expected files:
```
gbdt_train_val_loss.png
gbdt_feature_importance.png
gbdt_learning_rate_comparison.png
mlp_training_loss.png
mlp_depth_width_comparison.png
mlp_learning_rate_comparison.png
confusion_matrices.png
pr_curves.png
```

- [ ] **Step 2: Verify no data leakage**

Checklist:
- [ ] `StandardScaler` fit on `X_train` only (check `data_preparation.py:scale_features`)
- [ ] Mode imputation uses training modes only (check `data_preparation.py:impute_missing`)
- [ ] `encode_features` applies one-hot per split with alignment (check column alignment)
- [ ] Test set used only in final evaluation cells (cells 7-8)

- [ ] **Step 3: Clean up executed notebook**

Run: `rm -f notebook_executed.ipynb`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: end-to-end verification complete, all figures generated"
```
