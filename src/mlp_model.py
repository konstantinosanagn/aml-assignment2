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

    Explores: hidden_layer_sizes, activation, learning_rate_init, alpha, max_iter.
    """
    param_distributions = {
        "hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64), (128,), (256, 128)],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "alpha": [0.0001, 0.001, 0.01],
        "max_iter": [200, 500, 1000],
    }

    base_model = MLPClassifier(
        solver="adam",
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
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

    # Retrain best model to get loss_curve_
    best_model = train_mlp(X_train, y_train, params=search.best_params_)
    return best_model, search


def plot_training_loss_curve(model):
    """Plot MLP training loss curve and validation score.

    Adam optimizer should show fast AND smooth convergence (L7).
    Gap between training loss and validation score indicates overfitting (L4).
    """
    setup_plotting()
    has_val_scores = hasattr(model, "validation_scores_") and model.validation_scores_ is not None

    if has_val_scores:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Training loss
        axes[0].plot(model.loss_curve_, linewidth=2, color="coral")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].set_title("MLP: Training Loss Curve")
        # Validation score
        axes[1].plot(model.validation_scores_, linewidth=2, color="steelblue")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Validation Score")
        axes[1].set_title("MLP: Validation Score Over Training")
        plt.tight_layout()
    else:
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
        model = train_mlp(X_train, y_train, params={"hidden_layer_sizes": layers, "max_iter": 200})
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
        model = train_mlp(X_train, y_train, params={"learning_rate_init": lr, "max_iter": 200})
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
