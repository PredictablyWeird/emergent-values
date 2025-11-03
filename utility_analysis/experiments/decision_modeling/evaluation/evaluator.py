"""Evaluation functions for different prediction methods."""

import numpy as np
from typing import Dict, List, Tuple

from predictors import (
    evaluate_baseline,
    evaluate_baseline_discrete,
    load_exchange_rates,
    evaluate_exchange_rate_method,
    evaluate_exchange_rate_method_discrete,
    load_utility_curves,
    evaluate_log_utility_method,
    evaluate_log_utility_method_discrete,
)
from models import (
    train_mlp_with_cv,
    train_decision_tree_with_cv,
    create_mlp_classifier_pipeline,
    create_decision_tree_classifier_pipeline,
)
from evaluation.classification_cv import evaluate_classifier_with_cv
from data.preprocessing_discrete import get_label_distribution
from sklearn.preprocessing import LabelEncoder


def print_data_summary(
    X: np.ndarray,
    y: np.ndarray,
    features: List[str],
    metadata: List[Dict[str, str]]
) -> None:
    """Print summary information about the loaded data."""
    print(f"Loaded {len(X)} comparisons")
    print(f"Feature shape: {X.shape}")
    print(f"Features: {features}")
    print(f"Probability range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Sample metadata: {metadata[0] if metadata else 'None'}")


def evaluate_all_methods(
    X: np.ndarray,
    y: np.ndarray,
    metadata: List[Dict[str, str]],
    N_a: np.ndarray,
    N_b: np.ndarray,
    N_diff: np.ndarray,
    N_frac: np.ndarray,
    model_name: str,
    methods: List[str],
    subset_name: str = "all data",
    hidden_layer_sizes: Tuple[int, ...] = (5,),
    max_iter: int = 500,
    random_state: int = 42,
    alpha: float = 0.01,
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    log_utility_scale: float = 1.0,
    log_utility_method: str = "normal"
) -> None:
    """
    Evaluate prediction methods based on the methods list.
    
    Args:
        X: Feature matrix
        y: Target probabilities
        metadata: List of dictionaries with country information
        N_a: Array of N_a values
        N_b: Array of N_b values
        N_diff: Array of N_a - N_b differences
        N_frac: Array of N_b / N_a ratios
        model_name: Model name for loading exchange rates
        methods: List of method names to evaluate: ["baseline", "exchange_rates", "log_utility", "mlp", "decision_tree"]
        subset_name: Name of the subset being evaluated (for display)
        hidden_layer_sizes: Tuple of hidden layer sizes for MLP
        max_iter: Maximum number of iterations for MLP training
        random_state: Random state for reproducibility
        alpha: L2 regularization parameter for MLP
        max_depth: Maximum depth for Decision Tree
        min_samples_split: Minimum samples to split for Decision Tree
        min_samples_leaf: Minimum samples at leaf for Decision Tree
        log_utility_scale: Scaling factor for log utility sigmoid/normal method
        log_utility_method: "sigmoid" or "normal" for log utility predictions
    """
    print(f"\n{'='*60}")
    print(f"Evaluating methods on {subset_name}")
    print(f"{'='*60}")
    print(f"Number of comparisons: {len(X)}")
    print(f"Methods to evaluate: {methods}")
    print()
    
    # Evaluate baseline
    if "baseline" in methods:
        print("Evaluating baseline predictor (predicts 1 if N_a > N_b, 0 if N_a < N_b, 0.5 if N_a == N_b)...")
        baseline_results = evaluate_baseline(y, N_diff)
        
        print("\nBaseline results:")
        print(f"  R² score: {baseline_results['r2']:.4f}")
        print(f"  MSE: {baseline_results['mse']:.4f}")
        print(f"  MAE: {baseline_results['mae']:.4f}")
        print()
    
    # Try to load exchange rates and evaluate exchange rate method
    if "exchange_rates" in methods:
        print("Attempting to load exchange rates...")
        exchange_rates = load_exchange_rates(model_name)
        
        if exchange_rates:
            print(f"Loaded {len(exchange_rates)} exchange rate pairs")
            print("Evaluating exchange rate-based predictor...")
            exchange_results = evaluate_exchange_rate_method(
                y, metadata, exchange_rates, N_frac
            )
            
            print("\nExchange rate method results:")
            print(f"  R² score: {exchange_results['r2']:.4f}")
            print(f"  MSE: {exchange_results['mse']:.4f}")
            print(f"  MAE: {exchange_results['mae']:.4f}")
            print(f"  Coverage: {exchange_results['coverage']:.2%}")
            print()
        else:
            print("Exchange rate data not available - skipping exchange rate evaluation")
            print()
    
    # Try to load utility curves and evaluate log utility method
    if "log_utility" in methods:
        print("Attempting to load utility curves...")
        slopes, intercepts = load_utility_curves(model_name)
        
        if slopes and intercepts:
            print(f"Loaded utility curves for {len(slopes)} countries")
            print(f"Evaluating log utility-based predictor (using {log_utility_method} method, scale={log_utility_scale})...")
            log_utility_results = evaluate_log_utility_method(
                y, metadata, slopes, intercepts, N_a, N_b, 
                scale=log_utility_scale, method=log_utility_method
            )
            
            print("\nLog utility method results:")
            print(f"  R² score: {log_utility_results['r2']:.4f}")
            print(f"  MSE: {log_utility_results['mse']:.4f}")
            print(f"  MAE: {log_utility_results['mae']:.4f}")
            print(f"  Coverage: {log_utility_results['coverage']:.2%}")
            print()
        else:
            print("Utility curve data not available - skipping log utility evaluation")
            print()
    
    # Train MLP with cross-validation
    if "mlp" in methods:
        print(f"Training MLP regressor with cross-validation (L2 regularization, alpha={alpha})...")
        cv_results = train_mlp_with_cv(
            X, y, cv_folds=5,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            alpha=alpha
        )
        
        print(f"\nMLP Cross-validation results ({cv_results['cv_folds']}-fold CV):")
        print(f"Hidden layer sizes: {cv_results['hidden_layer_sizes']}")
        print(f"Alpha (L2 regularization): {cv_results['alpha']}")
        print()
        print("Test scores:")
        print(f"  R² score: {cv_results['test_r2_mean']:.4f} ± {cv_results['test_r2_std']:.4f}")
        print(f"  MSE: {cv_results['test_mse_mean']:.4f} ± {cv_results['test_mse_std']:.4f}")
        print(f"  MAE: {cv_results['test_mae_mean']:.4f} ± {cv_results['test_mae_std']:.4f}")
        print()
        print("Train scores:")
        print(f"  R² score: {cv_results['train_r2_mean']:.4f} ± {cv_results['train_r2_std']:.4f}")
        print(f"  MSE: {cv_results['train_mse_mean']:.4f} ± {cv_results['train_mse_std']:.4f}")
        print(f"  MAE: {cv_results['train_mae_mean']:.4f} ± {cv_results['train_mae_std']:.4f}")
        print()
    
    # Train Decision Tree with cross-validation
    if "decision_tree" in methods:
        print(f"Training Decision Tree regressor with cross-validation (max_depth={max_depth})...")
        dt_results = train_decision_tree_with_cv(
            X, y, cv_folds=5,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        
        print(f"\nDecision Tree Cross-validation results ({dt_results['cv_folds']}-fold CV):")
        print(f"Max depth: {dt_results['max_depth']}")
        print(f"Min samples split: {dt_results['min_samples_split']}")
        print(f"Min samples leaf: {dt_results['min_samples_leaf']}")
        print()
        print("Test scores:")
        print(f"  R² score: {dt_results['test_r2_mean']:.4f} ± {dt_results['test_r2_std']:.4f}")
        print(f"  MSE: {dt_results['test_mse_mean']:.4f} ± {dt_results['test_mse_std']:.4f}")
        print(f"  MAE: {dt_results['test_mae_mean']:.4f} ± {dt_results['test_mae_std']:.4f}")
        print()
        print("Train scores:")
        print(f"  R² score: {dt_results['train_r2_mean']:.4f} ± {dt_results['train_r2_std']:.4f}")
        print(f"  MSE: {dt_results['train_mse_mean']:.4f} ± {dt_results['train_mse_std']:.4f}")
        print(f"  MAE: {dt_results['train_mae_mean']:.4f} ± {dt_results['train_mae_std']:.4f}")
        print()


def evaluate_equal_n_subset(
    X: np.ndarray,
    y: np.ndarray,
    metadata: List[Dict[str, str]],
    N_a: np.ndarray,
    N_b: np.ndarray,
    N_diff: np.ndarray,
    N_frac: np.ndarray,
    model_name: str,
    methods: List[str],
    hidden_layer_sizes: Tuple[int, ...] = (5,),
    max_iter: int = 500,
    random_state: int = 42,
    alpha: float = 0.01,
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1
) -> None:
    """Evaluate methods on subset where N_a == N_b."""
    equal_n_mask = N_a == N_b
    if np.any(equal_n_mask):
        X_equal = X[equal_n_mask]
        y_equal = y[equal_n_mask]
        metadata_equal = [metadata[i] for i in range(len(metadata)) if equal_n_mask[i]]
        N_diff_equal = N_diff[equal_n_mask]
        N_frac_equal = N_frac[equal_n_mask]
        
        N_a_equal = N_a[equal_n_mask]
        N_b_equal = N_b[equal_n_mask]
        
        evaluate_all_methods(
            X_equal, y_equal, metadata_equal, N_a_equal, N_b_equal, 
            N_diff_equal, N_frac_equal, 
            model_name, methods, "N_a == N_b subset",
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            alpha=alpha,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
    else:
        print("\nNo comparisons with N_a == N_b found.")


def print_data_summary_discrete(
    X: np.ndarray,
    y: np.ndarray,
    y_labels: np.ndarray,
    features: List[str],
    metadata: List[Dict[str, str]],
    threshold: float = 0.1
) -> None:
    """Print summary information about the loaded data, including label distribution."""
    print(f"Loaded {len(X)} comparisons")
    print(f"Feature shape: {X.shape}")
    print(f"Features: {features}")
    print(f"Probability range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Threshold (t): {threshold}")
    print(f"Label distribution: {get_label_distribution(y_labels)}")
    print(f"Sample metadata: {metadata[0] if metadata else 'None'}")


def evaluate_all_methods_discrete(
    X: np.ndarray,
    y: np.ndarray,
    y_labels: np.ndarray,
    metadata: List[Dict[str, str]],
    N_a: np.ndarray,
    N_b: np.ndarray,
    N_diff: np.ndarray,
    N_frac: np.ndarray,
    model_name: str,
    methods: List[str],
    threshold: float = 0.1,
    subset_name: str = "all data",
    hidden_layer_sizes: Tuple[int, ...] = (5,),
    max_iter: int = 500,
    random_state: int = 42,
    alpha: float = 0.01,
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    log_utility_scale: float = 1.0,
    log_utility_method: str = "normal"
) -> None:
    """
    Evaluate prediction methods based on the methods list using discrete labels.
    
    Args:
        X: Feature matrix
        y: Target probabilities (continuous scores)
        y_labels: Target discrete labels
        metadata: List of dictionaries with country information
        N_a: Array of N_a values
        N_b: Array of N_b values
        N_diff: Array of N_a - N_b differences
        N_frac: Array of N_b / N_a ratios
        model_name: Model name for loading exchange rates
        methods: List of method names to evaluate: ["baseline", "exchange_rates", "log_utility", "mlp", "decision_tree"]
        threshold: Threshold parameter t for mapping scores to labels
        subset_name: Name of the subset being evaluated (for display)
        hidden_layer_sizes: Tuple of hidden layer sizes for MLP
        max_iter: Maximum number of iterations for MLP training
        random_state: Random state for reproducibility
        alpha: L2 regularization parameter for MLP
        max_depth: Maximum depth for Decision Tree
        min_samples_split: Minimum samples to split for Decision Tree
        min_samples_leaf: Minimum samples at leaf for Decision Tree
        log_utility_scale: Scaling factor for log utility sigmoid/normal method
        log_utility_method: "sigmoid" or "normal" for log utility predictions
    """
    print(f"\n{'='*60}")
    print(f"Evaluating methods on {subset_name}")
    print(f"{'='*60}")
    print(f"Number of comparisons: {len(X)}")
    print(f"Methods to evaluate: {methods}")
    print(f"Threshold (t): {threshold}")
    print(f"Label distribution: {get_label_distribution(y_labels)}")
    print()
    
    # Evaluate baseline
    if "baseline" in methods:
        print("Evaluating baseline predictor (predicts 1 if N_a > N_b, 0 if N_a < N_b, 0.5 if N_a == N_b)...")
        baseline_results = evaluate_baseline_discrete(y, N_diff, threshold)
        
        print("\nBaseline results:")
        print(f"  Accuracy: {baseline_results['accuracy']:.4f}")
        print("\nPer-class metrics:")
        for label in ['A', 'B', 'ambiguous']:
            print(f"  {label}:")
            print(f"    Precision: {baseline_results['precision'][label]:.4f}")
            print(f"    Recall: {baseline_results['recall'][label]:.4f}")
            print(f"    F1-score: {baseline_results['f1'][label]:.4f}")
            print(f"    Support: {baseline_results['support'][label]}")
        print("\nConfusion matrix:")
        print(f"  {baseline_results['confusion_matrix']}")
        print(f"\nTrue label distribution: {baseline_results['true_label_distribution']}")
        print(f"Predicted label distribution: {baseline_results['pred_label_distribution']}")
        print()
    
    # Try to load exchange rates and evaluate exchange rate method
    if "exchange_rates" in methods:
        print("Attempting to load exchange rates...")
        exchange_rates = load_exchange_rates(model_name)
        
        if exchange_rates:
            print(f"Loaded {len(exchange_rates)} exchange rate pairs")
            print("Evaluating exchange rate-based predictor (using exchange rates and N_frac)...")
            exchange_results = evaluate_exchange_rate_method_discrete(
                y, metadata, exchange_rates, N_frac, threshold
            )
            
            print("\nExchange rate method results:")
            print(f"  Accuracy: {exchange_results['accuracy']:.4f}")
            print(f"  Coverage: {exchange_results['coverage']:.2%}")
            print("\nPer-class metrics:")
            for label in ['A', 'B', 'ambiguous']:
                print(f"  {label}:")
                print(f"    Precision: {exchange_results['precision'][label]:.4f}")
                print(f"    Recall: {exchange_results['recall'][label]:.4f}")
                print(f"    F1-score: {exchange_results['f1'][label]:.4f}")
                print(f"    Support: {exchange_results['support'][label]}")
            print("\nConfusion matrix:")
            print(f"  {exchange_results['confusion_matrix']}")
            print(f"\nTrue label distribution: {exchange_results['true_label_distribution']}")
            print(f"Predicted label distribution: {exchange_results['pred_label_distribution']}")
            print()
        else:
            print("Exchange rate data not available - skipping exchange rate evaluation")
            print()
    
    # Try to load utility curves and evaluate log utility method
    if "log_utility" in methods:
        print("Attempting to load utility curves...")
        slopes, intercepts = load_utility_curves(model_name)
        
        if slopes and intercepts:
            print(f"Loaded utility curves for {len(slopes)} countries")
            print(f"Evaluating log utility-based predictor (using {log_utility_method} method, scale={log_utility_scale})...")
            log_utility_results = evaluate_log_utility_method_discrete(
                y, metadata, slopes, intercepts, N_a, N_b, threshold,
                scale=log_utility_scale, method=log_utility_method
            )
            
            print("\nLog utility method results:")
            print(f"  Accuracy: {log_utility_results['accuracy']:.4f}")
            print(f"  Coverage: {log_utility_results['coverage']:.2%}")
            print("\nPer-class metrics:")
            for label in ['A', 'B', 'ambiguous']:
                print(f"  {label}:")
                print(f"    Precision: {log_utility_results['precision'][label]:.4f}")
                print(f"    Recall: {log_utility_results['recall'][label]:.4f}")
                print(f"    F1-score: {log_utility_results['f1'][label]:.4f}")
                print(f"    Support: {log_utility_results['support'][label]}")
            print("\nConfusion matrix:")
            print(f"  {log_utility_results['confusion_matrix']}")
            print(f"\nTrue label distribution: {log_utility_results['true_label_distribution']}")
            print(f"Predicted label distribution: {log_utility_results['pred_label_distribution']}")
            print()
        else:
            print("Utility curve data not available - skipping log utility evaluation")
            print()
    
    # MLP classifier with 5-fold CV on discrete labels
    if "mlp" in methods:
        print(f"Evaluating MLP classifier with 5-fold CV on discrete labels (alpha={alpha})...")
        label_encoder = LabelEncoder()
        label_encoder.fit(y_labels)
        # Partition by country_x for CV
        groups = np.array([m["country_x"] for m in metadata])
        pipeline = create_mlp_classifier_pipeline(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            alpha=alpha
        )
        evaluate_classifier_with_cv(
            name="MLP",
            pipeline=pipeline,
            X=X,
            y_labels=y_labels,
            label_order=['A', 'B', 'ambiguous'],
            cv_folds=5,
            random_state=random_state,
            y_is_encoded=True,
            label_encoder=label_encoder,
            groups=groups
        )
    
    # Decision Tree classifier with 5-fold CV on discrete labels
    if "decision_tree" in methods:
        print(f"Evaluating Decision Tree classifier with 5-fold CV on discrete labels (max_depth={max_depth})...")
        pipeline = create_decision_tree_classifier_pipeline(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        # Partition by country_x for CV
        groups = np.array([m["country_x"] for m in metadata])
        evaluate_classifier_with_cv(
            name="Decision Tree",
            pipeline=pipeline,
            X=X,
            y_labels=y_labels,
            label_order=['A', 'B', 'ambiguous'],
            cv_folds=5,
            random_state=random_state,
            y_is_encoded=False,
            label_encoder=None,
            groups=groups
        )


def evaluate_equal_n_subset_discrete(
    X: np.ndarray,
    y: np.ndarray,
    y_labels: np.ndarray,
    metadata: List[Dict[str, str]],
    N_a: np.ndarray,
    N_b: np.ndarray,
    N_diff: np.ndarray,
    N_frac: np.ndarray,
    model_name: str,
    methods: List[str],
    threshold: float = 0.1,
    hidden_layer_sizes: Tuple[int, ...] = (5,),
    max_iter: int = 500,
    random_state: int = 42,
    alpha: float = 0.01,
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    log_utility_scale: float = 1.0,
    log_utility_method: str = "normal"
) -> None:
    """Evaluate methods on subset where N_a == N_b."""
    equal_n_mask = N_a == N_b
    if np.any(equal_n_mask):
        X_equal = X[equal_n_mask]
        y_equal = y[equal_n_mask]
        y_labels_equal = y_labels[equal_n_mask]
        metadata_equal = [metadata[i] for i in range(len(metadata)) if equal_n_mask[i]]
        N_diff_equal = N_diff[equal_n_mask]
        N_frac_equal = N_frac[equal_n_mask]
        
        N_a_equal = N_a[equal_n_mask]
        N_b_equal = N_b[equal_n_mask]
        
        evaluate_all_methods_discrete(
            X_equal, y_equal, y_labels_equal, metadata_equal, N_a_equal, N_b_equal, 
            N_diff_equal, N_frac_equal, 
            model_name, methods, threshold, "N_a == N_b subset",
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            alpha=alpha,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            log_utility_scale=log_utility_scale,
            log_utility_method=log_utility_method
        )
    else:
        print("\nNo comparisons with N_a == N_b found.")

