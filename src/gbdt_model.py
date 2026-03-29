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
