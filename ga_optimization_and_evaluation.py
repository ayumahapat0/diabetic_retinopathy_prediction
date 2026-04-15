import numpy as np
import random
import os
import json
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, roc_curve, auc
)
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# ============================================================================
# GA Optimization and Evaluation
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Project directory
BASE_DIR = "."

# Features directory
FEATURES_DIR = os.path.join(BASE_DIR, "features")

# Evalution Metrics and Plots directory 
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# Features
FEATURES_KEY = "features"

# Labels
LABELS_KEY = "labels"

# Feature Names
FEATURE_NAMES = [
    "contrast_64", "dissimilarity_64", "homogeneity_64",
    "energy_64", "correlation_64", "ASM_64",
    "GLCM_mean_64", "GLCM_var_64", "entropy_64",
    "contrast_128", "dissimilarity_128", "homogeneity_128",
    "energy_128", "correlation_128", "ASM_128",
    "GLCM_mean_128", "GLCM_var_128", "entropy_128",
]

# LightGBM parameter ranges for GA to search over
LIGHTGBM_PARAMS_RANGE = {
    "n_estimators"      : (50,   500),
    "max_depth"         : (3,    12),
    "min_child_weight"  : (1,    20),
    "learning_rate"     : (0.01, 0.3),
    "subsample"         : (0.5,  1.0),
    "colsample_bytree"  : (0.5,  1.0),
    "reg_alpha"         : (1.0,  10.0),
    "reg_lambda"        : (1.0,  10.0),
    "num_leaves"        : (20,   150),
    "min_child_samples" : (5,    50),
    "bagging_freq"      : (1,    10),
    "feature_fraction"  : (0.5,  1.0),
    "bagging_fraction"  : (0.5,  1.0),
}

# LightGBM Integer parameters
INTEGER_PARAMS = {"n_estimators", "max_depth", "min_child_weight",
                  "num_leaves", "min_child_samples", "bagging_freq"}


# json.dump chokes on numpy int64/float64, this handles it
class NumpyEncoder(json.JSONEncoder):
    
    def default(self, obj):
        
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


"""
Blended crossover utilizing randomized alpha and beta values (0, 1)
"""
def blended_crossover(parent1, parent2):

    diff = np.abs(parent1 - parent2)
    alpha, beta = np.random.rand(), np.random.rand()
    lo = np.minimum(parent1, parent2) - alpha * diff
    hi = np.maximum(parent1, parent2) + beta * diff
    child1 = np.clip(np.random.uniform(lo, hi), 0, 1)
    child2 = np.clip(np.random.uniform(lo, hi), 0, 1)
    return child1, child2

"""
Mutation with 20% probability as default
"""
def mutation(offspring, rate=0.20):

    for i in range(len(offspring)):
        if random.random() < rate:
            offspring[i] = np.random.rand()
    return offspring

"""
Tournament Selection with k = 3 as default
"""
def tournament_selection(pop, fits, k=3):
    idx = random.sample(range(len(pop)), k)
    winner = idx[np.argmin(fits[idx])]
    return pop[winner]

"""
Evolve with elitist strategy of 1 as default
"""
def evolve(pop, fits, mut_rate=0.20, k=3, n_elites=1):

    new_pop = []

    # Keep the best ones
    elite_idx = np.argsort(fits)[:n_elites]
    new_pop.extend(pop[elite_idx])

    while len(new_pop) < len(pop):

        p1 = tournament_selection(pop, fits, k)
        p2 = tournament_selection(pop, fits, k)
        c1, c2 = blended_crossover(p1, p2)
        c1 = mutation(c1, mut_rate)
        c2 = mutation(c2, mut_rate)
        new_pop.append(c1)
        if len(new_pop) < len(pop):
            new_pop.append(c2)

    return np.array(new_pop)


"""
Genetic Algorithm Optimizer class for 
feature selection and hyperparameter tuning
"""
class GA_Optimizer:
    # Paper used 0.6/0.4 but that was too aggressive on feature cutting
    # Bumped alpha to 0.8 so GA focuses more on accuracy
    ALPHA = 0.8
    BETA = 0.2
    
    # Minimum features to prevent overfitting (intuition based on paper)
    MIN_FEATURES = 3

    """
    Initialization
    """
    def __init__(self, pop_size=100, epochs=100, mut_rate=0.2, k=3, cv=5):
        self.pop_size = pop_size
        self.epochs = epochs
        self.mut_rate = mut_rate
        self.k = k
        self.cv = cv
        self.best_params = None
        self.selection_mask = None
        self.best_score = None
        self.history = []

    """
    Feature Selection Decoding
    [0, 0.5) = rejected features, [0.5, 1] = selected features
    """
    def decode_features(self, chrom):
        return (chrom >= 0.5) & (chrom <= 1)

    """
    Hyparameter Decoding
    """
    def decode_params(self, chrom):

        params = {}
        for i, (name, (lo, hi)) in enumerate(LIGHTGBM_PARAMS_RANGE.items()):
            val = chrom[i] * (hi - lo) + lo
            if name in INTEGER_PARAMS:
                val = int(round(val))
                val = np.clip(val, lo, hi)
            else:
                val = float(np.clip(val, lo, hi))
            params[name] = val
        return params
    
    """
    Fitness score to evaluate feature selection and hyperparameter values
    """
    def fitness(self, chrom, X, y):

        n_feat = X.shape[1]
        feat_mask = self.decode_features(chrom[:n_feat])
        X_sel = X[:, feat_mask]

        if X_sel.shape[1] < self.MIN_FEATURES:
            return 1  # penalty

        params = self.decode_params(chrom[n_feat:])
        model = lgb.LGBMClassifier(random_state=RANDOM_SEED, force_col_wise=True, verbose=-1, **params)
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=RANDOM_SEED)

        try:
            err = 1 - cross_val_score(model, X_sel, y, cv=skf, scoring='accuracy').mean()
        except ValueError:
            err = 1

        return self.ALPHA * err + self.BETA * (feat_mask.sum() / n_feat)

    """
    Function for Feature Selection and Hyperparameter Tuning
    """
    def fit(self, X, y):

        n_feat = X.shape[1]
        n_params = len(LIGHTGBM_PARAMS_RANGE)
        chrom_len = n_feat + n_params
        total_evals = self.pop_size * (1 + self.epochs)

        # try loading tqdm for progress bar
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_evals, desc="GA", unit="eval",
                        bar_format="{l_bar}{bar:30}{r_bar}")
        except ImportError:
            pbar = None
            print("  (pip install tqdm for a progress bar)")

        # init population randomly
        pop = np.random.rand(self.pop_size, chrom_len)
        fits = np.zeros(self.pop_size)

        for i in range(self.pop_size):
            fits[i] = self.fitness(pop[i], X, y)
            if pbar:
                pbar.update(1)
                pbar.set_postfix(epoch="init", best=f"{fits[:i+1].min():.4f}")

        best_chrom = pop[np.argmin(fits)].copy()
        best_fit = float(fits.min())
        self.history.append(best_fit)

        for ep in range(self.epochs):
            pop = evolve(pop, fits, self.mut_rate, self.k)
            for i in range(self.pop_size):
                fits[i] = self.fitness(pop[i], X, y)
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(epoch=f"{ep+1}/{self.epochs}", best=f"{best_fit:.4f}")

            curr_best = float(fits.min())
            if curr_best < best_fit:
                best_fit = curr_best
                best_chrom = pop[np.argmin(fits)].copy()

            self.history.append(best_fit)

            if not pbar:
                print(f"  Epoch {ep+1}/{self.epochs} | Best: {best_fit:.4f}")

        if pbar:
            pbar.close()

        self.selection_mask = self.decode_features(best_chrom[:n_feat])
        self.best_params = self.decode_params(best_chrom[n_feat:])
        self.best_score = best_fit
        return self

    """
    Get GA optimized model
    """
    def get_model(self):
        return lgb.LGBMClassifier(random_state=RANDOM_SEED, force_col_wise=True,
                                  verbose=-1, **self.best_params)


"""
Load the data from the npz files
"""
def load_data():

    train = np.load(os.path.join(FEATURES_DIR, "train_features.npz"))
    val   = np.load(os.path.join(FEATURES_DIR, "val_features.npz"))
    test  = np.load(os.path.join(FEATURES_DIR, "test_features.npz"))
    return (train[FEATURES_KEY], train[LABELS_KEY],
            val[FEATURES_KEY], val[LABELS_KEY],
            test[FEATURES_KEY], test[LABELS_KEY])

"""
Evaluate the model
"""
def eval_model(model, X, y, name):

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted')
    rec = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    cm = confusion_matrix(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Confusion Matrix: TN={cm[0][0]} FP={cm[0][1]} / FN={cm[1][0]} TP={cm[1][1]}")

    return {
        "accuracy": float(acc), "precision": float(prec),
        "recall": float(rec), "f1": float(f1), "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
    }

# Plotting

"""
Convergence Plot
"""
def save_convergence_plot(history, out_dir):

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(history)), history, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Best Fitness")
    plt.title("GA Convergence (BLX-alpha-beta)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence_plot.png"), dpi=150)
    plt.close()

"""
ROC Plot
"""
def save_roc_plot(baseline_res, ga_res, out_dir):

    plt.figure(figsize=(8, 8))
    plt.plot(baseline_res["fpr"], baseline_res["tpr"], 'b-', linewidth=2,
             label=f'Baseline (AUC={baseline_res["roc_auc"]:.4f})')
    plt.plot(ga_res["fpr"], ga_res["tpr"], 'r-', linewidth=2,
             label=f'GA-Optimized (AUC={ga_res["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close()

"""
Confusion Matrix Plot
"""
def save_confusion_plot(b_cm, ga_cm, n_sel, out_dir):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Healthy (0)", "DR (1)"]

    for ax, cm_data, title in [(axes[0], b_cm, "Baseline (18 features)"),
                                (axes[1], ga_cm, f"GA-Optimized ({n_sel} features)")]:
        cm = np.array(cm_data)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i][j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                        fontsize=18, fontweight="bold", color=color)
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Confusion Matrices (Test Set)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()

"""
Feature Selection Plot
"""
def save_feature_plot(mask, names, out_dir):

    colors = ["#2ecc71" if s else "#e74c3c" for s in mask]
    display = [("+ " + n if s else "- " + n) for n, s in zip(names, mask)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(names)), [1]*len(names), color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(display, fontsize=10)
    ax.set_xticks([])
    ax.set_title(f"Feature Selection ({int(sum(mask))}/{len(names)} kept)")
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#2ecc71", label="Selected"),
                       Patch(facecolor="#e74c3c", label="Rejected")], loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_selection.png"), dpi=150)
    plt.close()

"""
Baseline vs Optimized Evaluation Metrics Plot
"""
def save_comparison_plot(b_res, ga_res, n_sel, out_dir):

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    b_vals = [b_res[m] for m in metrics]
    g_vals = [ga_res[m] for m in metrics]
    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, b_vals, w, label="Baseline (18 features)", color="#3498db")
    bars2 = ax.bar(x + w/2, g_vals, w, label=f"GA-Optimized ({n_sel} features)", color="#e74c3c")
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs GA-Optimized (Test Set)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0.80, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metric_comparison.png"), dpi=150)
    plt.close()

def main():

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Class dist -> Train 0:{sum(y_train==0)} 1:{sum(y_train==1)} | "
          f"Val 0:{sum(y_val==0)} 1:{sum(y_val==1)} | "
          f"Test 0:{sum(y_test==0)} 1:{sum(y_test==1)}")

    # Baseline with all 18 features, default LightGBM
    print("\n" + "="*50)
    print("BASELINE (all features, default params)")
    print("="*50)

    baseline = lgb.LGBMClassifier(random_state=RANDOM_SEED, force_col_wise=True, verbose=-1)
    baseline.fit(X_train, y_train)

    b_val = eval_model(baseline, X_val, y_val, "Validation")
    b_test = eval_model(baseline, X_test, y_test, "Test")

    joblib.dump(baseline, os.path.join(OUTPUT_DIR, "baseline_model.joblib"))
    print(f"\n[SAVED] baseline model")


    # GA optimized LightGBM
    print("\n" + "="*50)
    print("GA-OPTIMIZED (BLX-alpha-beta, alpha=0.8 beta=0.2)")
    print("="*50)

    ga = GA_Optimizer(pop_size=75, epochs=25, mut_rate=0.2, k=3, cv=5)
    ga.fit(X_train, y_train)

    model = ga.get_model()
    model.fit(X_train[:, ga.selection_mask], y_train)

    ga_val = eval_model(model, X_val[:, ga.selection_mask], y_val, "Validation")
    ga_test = eval_model(model, X_test[:, ga.selection_mask], y_test, "Test")

    joblib.dump(model, os.path.join(OUTPUT_DIR, "ga_optimized_model.joblib"))
    np.savez(os.path.join(OUTPUT_DIR, "ga_state.npz"),
             selection_mask=ga.selection_mask,
             convergence_history=np.array(ga.history))
    print(f"\n[SAVED] GA model + state")


    # Which features were selected
    selected = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if ga.selection_mask[i]]
    rejected = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if not ga.selection_mask[i]]
    n_sel = int(sum(ga.selection_mask))

    print(f"\nFeatures selected ({n_sel}/{len(FEATURE_NAMES)}):")
    for f in selected:
        print(f"  + {f}")
    print(f"Features rejected ({len(rejected)}):")
    for f in rejected:
        print(f"  - {f}")
    print(f"Reduction: {len(rejected)/len(FEATURE_NAMES)*100:.1f}%")


    # Best params found
    print(f"\nBest hyperparams:")
    for p, v in ga.best_params.items():
        print(f"  {p}: {v}")
    print(f"Best fitness: {ga.best_score:.4f}")


    # Convergence log
    print(f"\nConvergence:")
    for i, f in enumerate(ga.history):
        print(f"  epoch {i}: {f:.4f}")


    # Comparison
    print(f"\nBaseline vs GA (test set):")
    print(f"  {'metric':<10} {'baseline':>10} {'GA':>10} {'diff':>10}")
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        b = b_test[m]
        g = ga_test[m]
        d = g - b
        print(f"  {m:<10} {b:>10.4f} {g:>10.4f} {d:>+10.4f}")
    print(f"  features:       18         {n_sel}")


    # Save all plots
    print("\nSaving plots...")
    save_convergence_plot(ga.history, OUTPUT_DIR)
    save_roc_plot(b_test, ga_test, OUTPUT_DIR)
    save_confusion_plot(b_test["confusion_matrix"], ga_test["confusion_matrix"], n_sel, OUTPUT_DIR)
    save_feature_plot(ga.selection_mask, FEATURE_NAMES, OUTPUT_DIR)
    save_comparison_plot(b_test, ga_test, n_sel, OUTPUT_DIR)
    print("Done.")


    # Dump everything to json so we don't lose numbers
    results = {
        "ga_config": {
            "pop_size": ga.pop_size, "epochs": ga.epochs,
            "mutation_prob": ga.mut_rate, "tournament_size": ga.k,
            "cv_folds": ga.cv, "alpha": ga.ALPHA, "beta": ga.BETA,
        },
        "baseline_val": b_val, "baseline_test": b_test,
        "ga_val": ga_val, "ga_test": ga_test,
        "selected_features": selected, "rejected_features": rejected,
        "n_selected": n_sel,
        "best_params": ga.best_params,
        "best_fitness": ga.best_score,
        "convergence_history": ga.history,
        "feature_reduction_pct": len(rejected) / len(FEATURE_NAMES) * 100,
    }

    with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\n[DONE] Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()

