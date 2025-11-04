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
from models.mlp_model import MLPModel
from models.decision_tree_model import DecisionTreeModel
from models.exchange_rates_model import ExchangeRatesModel
from models.log_utility_model import LogUtilityModel
from evaluation.cv_utils import create_cv_splits
from evaluation.model_evaluator import evaluate_model_with_cv
from evaluation.classifier_evaluator import evaluate_classifier_with_cv as evaluate_classifier_model_with_cv
from data.preprocessing_discrete import get_label_distribution


def _print_regression_results(results: Dict, method_name: str) -> None:
    """Helper function to print regression results (R², MSE, MAE)."""
    print(f"\n{method_name} results:")
    print(f"  R² score: {results['r2']:.4f}")
    print(f"  MSE: {results['mse']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")
    if 'coverage' in results:
        print(f"  Coverage: {results['coverage']:.2%}")
    print()


def _print_cv_regression_results(results: Dict, method_name: str) -> None:
    """Helper function to print CV regression results."""
    print(f"\n{method_name} Cross-validation results ({results['cv_folds']}-fold CV):")
    
    # Print hyperparameters if present
    if 'hidden_layer_sizes' in results:
        print(f"Hidden layer sizes: {results['hidden_layer_sizes']}")
    if 'alpha' in results:
        print(f"Alpha (L2 regularization): {results['alpha']}")
    if 'max_depth' in results:
        print(f"Max depth: {results['max_depth']}")
    if 'min_samples_split' in results:
        print(f"Min samples split: {results['min_samples_split']}")
    if 'min_samples_leaf' in results:
        print(f"Min samples leaf: {results['min_samples_leaf']}")
    if 'scale' in results:
        print(f"Scale: {results['scale']}")
    if 'method' in results:
        print(f"Method: {results['method']}")
    if 'tolerance' in results:
        print(f"Tolerance: {results['tolerance']}")
    print()
    
    print("Test scores:")
    print(f"  R² score: {results['test_r2_mean']:.4f} ± {results['test_r2_std']:.4f}")
    print(f"  MSE: {results['test_mse_mean']:.4f} ± {results['test_mse_std']:.4f}")
    print(f"  MAE: {results['test_mae_mean']:.4f} ± {results['test_mae_std']:.4f}")
    print()
    print("Train scores:")
    print(f"  R² score: {results['train_r2_mean']:.4f} ± {results['train_r2_std']:.4f}")
    print(f"  MSE: {results['train_mse_mean']:.4f} ± {results['train_mse_std']:.4f}")
    print(f"  MAE: {results['train_mae_mean']:.4f} ± {results['train_mae_std']:.4f}")
    print()


def _print_classification_results(results: Dict, method_name: str) -> None:
    """Helper function to print classification results."""
    print(f"\n{method_name} results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    if 'coverage' in results:
        print(f"  Coverage: {results['coverage']:.2%}")
    print("\nPer-class metrics:")
    for label in ['A', 'B', 'ambiguous']:
        print(f"  {label}:")
        print(f"    Precision: {results['precision'][label]:.4f}")
        print(f"    Recall: {results['recall'][label]:.4f}")
        print(f"    F1-score: {results['f1'][label]:.4f}")
        print(f"    Support: {results['support'][label]}")
    print("\nConfusion matrix:")
    print(f"  {results['confusion_matrix']}")
    print(f"\nTrue label distribution: {results['true_label_distribution']}")
    print(f"Predicted label distribution: {results['pred_label_distribution']}")
    print()


def _print_cv_classification_results(results: Dict, method_name: str) -> None:
    """Helper function to print CV classification results."""
    print(f"\n{method_name} Cross-validation results ({results['cv_folds']}-fold CV):")
    
    # Print hyperparameters if present
    if 'hidden_layer_sizes' in results:
        print(f"Hidden layer sizes: {results['hidden_layer_sizes']}")
    if 'alpha' in results:
        print(f"Alpha (L2 regularization): {results['alpha']}")
    if 'max_depth' in results:
        print(f"Max depth: {results['max_depth']}")
    if 'min_samples_split' in results:
        print(f"Min samples split: {results['min_samples_split']}")
    if 'min_samples_leaf' in results:
        print(f"Min samples leaf: {results['min_samples_leaf']}")
    if 'scale' in results:
        print(f"Scale: {results['scale']}")
    if 'method' in results:
        print(f"Method: {results['method']}")
    if 'tolerance' in results:
        print(f"Tolerance: {results['tolerance']}")
    print()
    
    print("Test scores:")
    print(f"  Accuracy: {results['test_accuracy_mean']:.4f} ± {results['test_accuracy_std']:.4f}")
    print("\nPer-class metrics (mean ± std across folds):")
    for label in ['A', 'B', 'ambiguous']:
        print(f"  {label}:")
        print(f"    Precision: {results['test_precision_mean'][label]:.4f} ± {results['test_precision_std'][label]:.4f}")
        print(f"    Recall:    {results['test_recall_mean'][label]:.4f} ± {results['test_recall_std'][label]:.4f}")
        print(f"    F1-score:  {results['test_f1_mean'][label]:.4f} ± {results['test_f1_std'][label]:.4f}")
    
    if 'train_accuracy_mean' in results:
        print("\nTrain scores:")
        print(f"  Accuracy: {results['train_accuracy_mean']:.4f} ± {results['train_accuracy_std']:.4f}")
        print("\nPer-class metrics (mean ± std across folds):")
        for label in ['A', 'B', 'ambiguous']:
            print(f"  {label}:")
            print(f"    Precision: {results['train_precision_mean'][label]:.4f} ± {results['train_precision_std'][label]:.4f}")
            print(f"    Recall:    {results['train_recall_mean'][label]:.4f} ± {results['train_recall_std'][label]:.4f}")
            print(f"    F1-score:  {results['train_f1_mean'][label]:.4f} ± {results['train_f1_std'][label]:.4f}")
    
    print("\nOut-of-fold confusion matrix:")
    print(f"  {results['confusion_matrix_oof']}")
    print(f"\nTrue label distribution: {results['true_label_distribution']}")
    print(f"Predicted label distribution (OOF): {results['pred_label_distribution_oof']}")
    print()


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
        _print_regression_results(baseline_results, "Baseline")
    
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
            _print_regression_results(exchange_results, "Exchange rate method")
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
            _print_regression_results(log_utility_results, "Log utility method")
        else:
            print("Utility curve data not available - skipping log utility evaluation")
            print()
    
    # Create CV splits once globally (for methods that use CV)
    cv_folds = 5
    cv_splits = create_cv_splits(X, y, cv_folds=cv_folds, random_state=random_state)
    
    # Train exchange rate model with cross-validation (fits utility curves from training data)
    if "exchange_rates_cv" in methods:
        print(f"Training exchange rate model with cross-validation (scale={log_utility_scale}, method={log_utility_method})...")
        exchange_model = ExchangeRatesModel(
            scale=log_utility_scale,
            method=log_utility_method,
            tolerance=0.28
        )
        exchange_cv_results = evaluate_model_with_cv(
            exchange_model,
            X, y, cv_splits,
            metadata=metadata,
            N_a=N_a,
            N_b=N_b,
            N_frac=N_frac
        )
        _print_cv_regression_results(exchange_cv_results, "Exchange Rate")
    
    # Train log utility model with cross-validation (fits utility curves from training data)
    if "log_utility_cv" in methods:
        print(f"Training log utility model with cross-validation (scale={log_utility_scale}, method={log_utility_method})...")
        log_utility_model = LogUtilityModel(
            scale=log_utility_scale,
            method=log_utility_method
        )
        log_utility_cv_results = evaluate_model_with_cv(
            log_utility_model,
            X, y, cv_splits,
            metadata=metadata,
            N_a=N_a,
            N_b=N_b
        )
        _print_cv_regression_results(log_utility_cv_results, "Log Utility")
    
    # Train MLP with cross-validation
    if "mlp" in methods:
        print(f"Training MLP regressor with cross-validation (L2 regularization, alpha={alpha})...")
        mlp_model = MLPModel(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            alpha=alpha
        )
        mlp_results = evaluate_model_with_cv(
            mlp_model,
            X, y, cv_splits
        )
        _print_cv_regression_results(mlp_results, "MLP")
    
    # Train Decision Tree with cross-validation
    if "decision_tree" in methods:
        print(f"Training Decision Tree regressor with cross-validation (max_depth={max_depth})...")
        dt_model = DecisionTreeModel(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        dt_results = evaluate_model_with_cv(
            dt_model,
            X, y, cv_splits
        )
        _print_cv_regression_results(dt_results, "Decision Tree")


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
    log_utility_method: str = "normal",
    optimize_hyperparameters: bool = True
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
        _print_classification_results(baseline_results, "Baseline")
    
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
            _print_classification_results(exchange_results, "Exchange rate method")
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
            _print_classification_results(log_utility_results, "Log utility method")
        else:
            print("Utility curve data not available - skipping log utility evaluation")
            print()
    
    # Create CV splits once globally (for methods that use CV)
    # Use GroupKFold to ensure all rows with the same country stay in the same fold
    cv_folds = 5
    groups = np.array([sorted([m["country_x"], m["country_y"]])[0] for m in metadata]) if metadata else None
    cv_splits = create_cv_splits(X, y_labels, cv_folds=cv_folds, random_state=random_state, groups=groups)
    
    # Train exchange rate model with cross-validation (fits utility curves from training data)
    if "exchange_rates_cv" in methods:
        print(f"Training exchange rate model with cross-validation (scale={log_utility_scale}, method={log_utility_method}, optimize_tolerance={optimize_hyperparameters})...")
        exchange_model = ExchangeRatesModel(
            scale=log_utility_scale,
            method=log_utility_method,
            tolerance=0.28,
            optimize_tolerance=optimize_hyperparameters
        )
        # Use classifier evaluation since we're in discrete mode
        # These models need probabilities (y) for training, but labels (y_labels) for evaluation
        exchange_cv_results = evaluate_classifier_model_with_cv(
            exchange_model,
            X, y_labels, cv_splits,
            label_order=['A', 'B', 'ambiguous'],
            metadata=metadata,
            N_a=N_a,
            N_b=N_b,
            N_frac=N_frac,
            y_probs=y,  # Pass probabilities for training
            threshold=threshold
        )
        _print_cv_classification_results(exchange_cv_results, "Exchange Rate")
    
    # Train log utility model with cross-validation (fits utility curves from training data)
    if "log_utility_cv" in methods:
        print(f"Training log utility model with cross-validation (scale={log_utility_scale}, method={log_utility_method}, optimize_scale={optimize_hyperparameters})...")
        log_utility_model = LogUtilityModel(
            scale=log_utility_scale,
            method=log_utility_method,
            optimize_scale=optimize_hyperparameters
        )
        # Use classifier evaluation since we're in discrete mode
        # These models need probabilities (y) for training, but labels (y_labels) for evaluation
        log_utility_cv_results = evaluate_classifier_model_with_cv(
            log_utility_model,
            X, y_labels, cv_splits,
            label_order=['A', 'B', 'ambiguous'],
            metadata=metadata,
            N_a=N_a,
            N_b=N_b,
            y_probs=y,  # Pass probabilities for training
            threshold=threshold
        )
        _print_cv_classification_results(log_utility_cv_results, "Log Utility")
    
    # MLP classifier with 5-fold CV on discrete labels
    if "mlp" in methods:
        print(f"Evaluating MLP classifier with 5-fold CV on discrete labels (alpha={alpha})...")
        mlp_model = MLPModel(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            alpha=alpha,
            task="classification"
        )
        mlp_results = evaluate_classifier_model_with_cv(
            mlp_model,
            X, y_labels, cv_splits,
            label_order=['A', 'B', 'ambiguous']
        )
        _print_cv_classification_results(mlp_results, "MLP")
    
    # Decision Tree classifier with 5-fold CV on discrete labels
    if "decision_tree" in methods:
        print(f"Evaluating Decision Tree classifier with 5-fold CV on discrete labels (max_depth={max_depth})...")
        dt_model = DecisionTreeModel(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            task="classification"
        )
        dt_results = evaluate_classifier_model_with_cv(
            dt_model,
            X, y_labels, cv_splits,
            label_order=['A', 'B', 'ambiguous']
        )
        _print_cv_classification_results(dt_results, "Decision Tree")


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

