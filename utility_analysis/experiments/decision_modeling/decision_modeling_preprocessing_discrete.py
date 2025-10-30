import argparse
import numpy as np
from typing import Dict, List, Tuple
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from collections import Counter

from decision_modeling_preprocessing import (
    compute_log_utility_predictions,
    compute_exchange_rate_predictions,
    load_country_features,
    load_decision_file,
    create_feature_difference_vector,
    load_exchange_rates,
    load_utility_curves,
    compute_baseline_predictions,
)


def map_scores_to_labels(scores: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Map continuous scores to discrete labels.
    
    Args:
        scores: Array of continuous scores (probabilities)
        threshold: Threshold parameter t for mapping
                   - "A" if score > 0.5 + t
                   - "B" if score < 0.5 - t
                   - "ambiguous" otherwise
    
    Returns:
        Array of discrete labels: "A", "B", or "ambiguous"
    """
    labels = np.empty(len(scores), dtype=object)
    high_threshold = 0.5 + threshold
    low_threshold = 0.5 - threshold
    
    labels[scores > high_threshold] = "A"
    labels[scores < low_threshold] = "B"
    labels[(scores >= low_threshold) & (scores <= high_threshold)] = "ambiguous"
    
    return labels


def get_label_distribution(labels: np.ndarray) -> Dict[str, int]:
    """
    Get distribution of labels.
    
    Args:
        labels: Array of discrete labels
    
    Returns:
        Dictionary mapping label to count
    """
    return dict(Counter(labels))


def evaluate_classifier_with_cv(
    name: str,
    pipeline: Pipeline,
    X: np.ndarray,
    y_labels: np.ndarray,
    label_order: List[str],
    cv_folds: int = 5,
    random_state: int = 42,
    y_is_encoded: bool = False,
    label_encoder: LabelEncoder = None
) -> None:
    """
    Evaluate a classifier using Stratified K-Fold cross-validation.
    Prints mean±std metrics across folds and the out-of-fold confusion matrix.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    per_fold_metrics = []  # list of dicts with accuracy, precision/recall/f1 per-class (macro for summary)
    oof_pred_labels = np.empty_like(y_labels, dtype=object)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_labels)):
        model = clone(pipeline)
        if y_is_encoded:
            # Encode using provided encoder
            y_train_enc = label_encoder.transform(y_labels[train_idx])
            model.fit(X[train_idx], y_train_enc)
            y_val_pred_enc = model.predict(X[val_idx])
            y_val_pred = label_encoder.inverse_transform(y_val_pred_enc)
        else:
            model.fit(X[train_idx], y_labels[train_idx])
            y_val_pred = model.predict(X[val_idx])

        y_true_val = y_labels[val_idx]
        oof_pred_labels[val_idx] = y_val_pred

        accuracy = accuracy_score(y_true_val, y_val_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_val, y_val_pred, labels=label_order, zero_division=0
        )

        per_fold_metrics.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        })

    # Aggregate per-fold metrics (means and stds)
    accuracies = np.array([m['accuracy'] for m in per_fold_metrics])
    precisions = np.stack([m['precision'] for m in per_fold_metrics], axis=0)
    recalls = np.stack([m['recall'] for m in per_fold_metrics], axis=0)
    f1s = np.stack([m['f1'] for m in per_fold_metrics], axis=0)

    print(f"\n{name} (5-fold CV) results:")
    print(f"  Accuracy: {accuracies.mean():.4f} ± {accuracies.std():.4f}")

    print("\nPer-class metrics (mean ± std across folds):")
    for i, label in enumerate(label_order):
        print(f"  {label}:")
        print(f"    Precision: {precisions[:, i].mean():.4f} ± {precisions[:, i].std():.4f}")
        print(f"    Recall:    {recalls[:, i].mean():.4f} ± {recalls[:, i].std():.4f}")
        print(f"    F1-score:  {f1s[:, i].mean():.4f} ± {f1s[:, i].std():.4f}")

    # Out-of-fold confusion matrix and distributions
    cm_oof = confusion_matrix(y_labels, oof_pred_labels, labels=label_order)
    print("\nOut-of-fold confusion matrix:")
    print(f"  {cm_oof}")
    print(f"\nTrue label distribution: {get_label_distribution(y_labels)}")
    print(f"Predicted label distribution (OOF): {get_label_distribution(oof_pred_labels)}")


 


def preprocess_decision_data(
    csv_path: str,
    jsonl_path: str,
    threshold: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[Dict[str, str]], np.ndarray, np.ndarray]:
    """
    Preprocess decision data from CSV and country features from JSONL.
    Maps scores to discrete labels using threshold.
    
    Args:
        csv_path: Path to CSV file with country comparisons
        jsonl_path: Path to JSONL file with country features
        threshold: Threshold parameter t for mapping scores to labels
        
    Returns:
        Tuple of:
            - X: numpy array with input features (differences)
            - y: numpy array with probabilities (original scores)
            - y_labels: numpy array with discrete labels ("A", "B", "ambiguous")
            - features: list of feature names
            - metadata: list of dictionaries with "country_a" and "country_b" keys
            - N_a: numpy array with N_a values
            - N_b: numpy array with N_b values
    """
    # Load country features
    country_features = load_country_features(jsonl_path)
    
    # Get feature names from the first country (assuming all have same features)
    if not country_features:
        raise ValueError("No country features found in JSONL file")
    
    first_country = next(iter(country_features))
    feature_names = sorted(country_features[first_country].keys())
    
    # Add N_diff and N_frac feature names (will be appended at the end)
    feature_names = feature_names + ['N_diff', 'N_frac']
    
    # Load decision data
    decisions = load_decision_file(csv_path)
    
    # Build feature matrices and metadata
    X_rows = []
    y_values = []
    metadata_list = []
    N_a_values = []
    N_b_values = []
    
    missing_countries = set()
    
    for N_a, country_a, N_b, country_b, probability in decisions:
        # Check if both countries have features
        if country_a not in country_features:
            missing_countries.add(country_a)
            continue
        if country_b not in country_features:
            missing_countries.add(country_b)
            continue
        
        # Get feature vectors
        features_a = country_features[country_a]
        features_b = country_features[country_b]
        
        # Create difference vector for country features
        diff_vector = create_feature_difference_vector(
            features_a, features_b, feature_names[:-2]  # Exclude N_diff and N_frac from feature names here
        )
        
        # Calculate N_diff = N_a - N_b and N_frac = N_b / N_a
        N_diff = N_a - N_b
        N_frac = N_b / N_a if N_a != 0 else 0.0
        
        # Append N_diff and N_frac to the feature vector
        feature_vector = np.append(diff_vector, [N_diff, N_frac])
        
        X_rows.append(feature_vector)
        y_values.append(probability)
        metadata_list.append({
            "country_a": country_a,
            "country_b": country_b
        })
        N_a_values.append(N_a)
        N_b_values.append(N_b)
    
    if missing_countries:
        print(f"Warning: Missing features for countries: {missing_countries}")
    
    # Convert to numpy arrays
    X = np.array(X_rows)
    y = np.array(y_values)
    N_a = np.array(N_a_values)
    N_b = np.array(N_b_values)
    
    # Map scores to discrete labels
    y_labels = map_scores_to_labels(y, threshold)
    
    return X, y, y_labels, feature_names, metadata_list, N_a, N_b


def create_mlp_pipeline(
    hidden_layer_sizes: Tuple[int, ...] = (5,),
    max_iter: int = 500,
    random_state: int = 42,
    alpha: float = 0.01
) -> Pipeline:
    """
    Create MLP pipeline with scaling and MLP classifier.
    
    Args:
        hidden_layer_sizes: Tuple of hidden layer sizes
        max_iter: Maximum number of iterations
        random_state: Random state for reproducibility
        alpha: L2 regularization parameter (higher = more sparsity/regularization)
        
    Returns:
        Pipeline with scaler and MLP classifier
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            activation='tanh',
            validation_fraction=0.1,
            alpha=alpha
        )),
    ])


def create_decision_tree_pipeline(
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> Pipeline:
    """
    Create Decision Tree pipeline with scaling and Decision Tree classifier.
    
    Args:
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum number of samples required to split a node
        min_samples_leaf: Minimum number of samples required at a leaf node
        random_state: Random state for reproducibility
        
    Returns:
        Pipeline with scaler and Decision Tree classifier
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('dt', DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )),
    ])


def train_decision_tree_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> Dict:
    """
    Train Decision Tree regressor using cross-validation to predict probabilities from features.
    
    Args:
        X: Input features array
        y: Target probabilities
        cv_folds: Number of cross-validation folds
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum number of samples required to split a node
        min_samples_leaf: Minimum number of samples required at a leaf node
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with cross-validation results including scores and metrics
    """
    pipeline = create_decision_tree_pipeline(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    # Perform cross-validation
    scoring = {
        'r2': 'r2',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'neg_mean_absolute_error': 'neg_mean_absolute_error'
    }
    
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv_folds,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Calculate mean and std for each metric
    results = {
        'cv_folds': cv_folds,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'test_r2_mean': np.mean(cv_results['test_r2']),
        'test_r2_std': np.std(cv_results['test_r2']),
        'test_mse_mean': -np.mean(cv_results['test_neg_mean_squared_error']),
        'test_mse_std': np.std(cv_results['test_neg_mean_squared_error']),
        'test_mae_mean': -np.mean(cv_results['test_neg_mean_absolute_error']),
        'test_mae_std': np.std(cv_results['test_neg_mean_absolute_error']),
        'train_r2_mean': np.mean(cv_results['train_r2']),
        'train_r2_std': np.std(cv_results['train_r2']),
        'train_mse_mean': -np.mean(cv_results['train_neg_mean_squared_error']),
        'train_mse_std': np.std(cv_results['train_neg_mean_squared_error']),
        'train_mae_mean': -np.mean(cv_results['train_neg_mean_absolute_error']),
        'train_mae_std': np.std(cv_results['train_neg_mean_absolute_error']),
    }
    
    return results


def train_mlp_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    hidden_layer_sizes: Tuple[int, ...] = (5,),
    max_iter: int = 500,
    random_state: int = 42,
    alpha: float = 0.01
) -> Dict:
    """
    Train MLP regressor using cross-validation to predict probabilities from features.
    
    Args:
        X: Input features array
        y: Target probabilities
        cv_folds: Number of cross-validation folds
        hidden_layer_sizes: Tuple of hidden layer sizes
        max_iter: Maximum number of iterations
        random_state: Random state for reproducibility
        alpha: L2 regularization parameter (higher = more sparsity/regularization)
        
    Returns:
        Dictionary with cross-validation results including scores and metrics
    """
    # Create pipeline with scaling and MLP
    # Scaling is important for neural networks
    # alpha parameter adds L2 regularization to encourage sparsity of weights
    pipeline = create_mlp_pipeline(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        alpha=alpha
    )
    
    # Perform cross-validation
    # Using multiple scoring metrics for regression
    scoring = {
        'r2': 'r2',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'neg_mean_absolute_error': 'neg_mean_absolute_error'
    }
    
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv_folds,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Calculate mean and std for each metric
    results = {
        'cv_folds': cv_folds,
        'hidden_layer_sizes': hidden_layer_sizes,
        'alpha': alpha,
        'test_r2_mean': np.mean(cv_results['test_r2']),
        'test_r2_std': np.std(cv_results['test_r2']),
        'test_mse_mean': -np.mean(cv_results['test_neg_mean_squared_error']),
        'test_mse_std': np.std(cv_results['test_neg_mean_squared_error']),
        'test_mae_mean': -np.mean(cv_results['test_neg_mean_absolute_error']),
        'test_mae_std': np.std(cv_results['test_neg_mean_absolute_error']),
        'train_r2_mean': np.mean(cv_results['train_r2']),
        'train_r2_std': np.std(cv_results['train_r2']),
        'train_mse_mean': -np.mean(cv_results['train_neg_mean_squared_error']),
        'train_mse_std': np.std(cv_results['train_neg_mean_squared_error']),
        'train_mae_mean': -np.mean(cv_results['train_neg_mean_absolute_error']),
        'train_mae_std': np.std(cv_results['train_neg_mean_absolute_error']),
    }
    
    return results


 


def evaluate_baseline(y_true: np.ndarray, N_diff: np.ndarray, threshold: float = 0.1) -> Dict:
    """
    Evaluate baseline predictor performance using discrete label metrics.
    
    Args:
        y_true: True probability values
        N_diff: Array of N_a - N_b differences
        threshold: Threshold parameter t for mapping scores to labels
        
    Returns:
        Dictionary with classification metrics
    """
    y_pred = compute_baseline_predictions(N_diff)
    
    # Map to discrete labels
    y_true_labels = map_scores_to_labels(y_true, threshold)
    y_pred_labels = map_scores_to_labels(y_pred, threshold)
    
    # Classification metrics
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, labels=['A', 'B', 'ambiguous'], zero_division=0
    )
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=['A', 'B', 'ambiguous'])
    
    results = {
        'accuracy': accuracy,
        'precision': dict(zip(['A', 'B', 'ambiguous'], precision)),
        'recall': dict(zip(['A', 'B', 'ambiguous'], recall)),
        'f1': dict(zip(['A', 'B', 'ambiguous'], f1)),
        'support': dict(zip(['A', 'B', 'ambiguous'], support)),
        'confusion_matrix': cm,
        'true_label_distribution': get_label_distribution(y_true_labels),
        'pred_label_distribution': get_label_distribution(y_pred_labels)
    }
    
    return results


 


def evaluate_exchange_rate_method(
    y_true: np.ndarray,
    metadata: List[Dict[str, str]],
    exchange_rates: Dict[Tuple[str, str], float],
    N_frac: np.ndarray,
    threshold: float = 0.1
) -> Dict:
    """
    Evaluate exchange rate-based predictor performance using discrete label metrics.
    
    Args:
        y_true: True probability values
        metadata: List of dictionaries with 'country_a' and 'country_b' keys
        exchange_rates: Dictionary mapping (country_a, country_b) to exchange rate
        N_frac: Array of N_b / N_a ratios
        threshold: Threshold parameter t for mapping scores to labels
        
    Returns:
        Dictionary with classification metrics
    """
    y_pred = compute_exchange_rate_predictions(metadata, exchange_rates, N_frac)
    
    # Map to discrete labels
    y_true_labels = map_scores_to_labels(y_true, threshold)
    y_pred_labels = map_scores_to_labels(y_pred, threshold)
    
    # Classification metrics
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, labels=['A', 'B', 'ambiguous'], zero_division=0
    )
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=['A', 'B', 'ambiguous'])
    
    results = {
        'accuracy': accuracy,
        'precision': dict(zip(['A', 'B', 'ambiguous'], precision)),
        'recall': dict(zip(['A', 'B', 'ambiguous'], recall)),
        'f1': dict(zip(['A', 'B', 'ambiguous'], f1)),
        'support': dict(zip(['A', 'B', 'ambiguous'], support)),
        'confusion_matrix': cm,
        'coverage': np.mean([(meta['country_a'], meta['country_b']) in exchange_rates for meta in metadata]),
        'true_label_distribution': get_label_distribution(y_true_labels),
        'pred_label_distribution': get_label_distribution(y_pred_labels)
    }
    
    return results


 


def evaluate_log_utility_method(
    y_true: np.ndarray,
    metadata: List[Dict[str, str]],
    slopes: Dict[str, float],
    intercepts: Dict[str, float],
    N_a: np.ndarray,
    N_b: np.ndarray,
    threshold: float = 0.1,
    scale: float = 1.0,
    method: str = "normal"
) -> Dict:
    """
    Evaluate log utility-based predictor performance using discrete label metrics.
    
    Args:
        y_true: True probability values
        metadata: List of dictionaries with 'country_a' and 'country_b' keys
        slopes: Dictionary mapping country names to their utility curve slopes
        intercepts: Dictionary mapping country names to their utility curve intercepts
        N_a: Array of N_a values
        N_b: Array of N_b values
        threshold: Threshold parameter t for mapping scores to labels
        scale: Scaling factor for sigmoid (higher = sharper transition) or std dev for normal method
        method: "sigmoid" (logistic) or "normal" (probit/Thurstonian)
        
    Returns:
        Dictionary with classification metrics
    """
    y_pred = compute_log_utility_predictions(metadata, slopes, intercepts, N_a, N_b, scale=scale, method=method)
    
    # Map to discrete labels
    y_true_labels = map_scores_to_labels(y_true, threshold)
    y_pred_labels = map_scores_to_labels(y_pred, threshold)
    
    # Classification metrics
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, labels=['A', 'B', 'ambiguous'], zero_division=0
    )
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=['A', 'B', 'ambiguous'])
    
    # Calculate coverage: fraction of comparisons where both countries have utility curves
    coverage = np.mean([
        (meta['country_a'] in slopes and meta['country_a'] in intercepts and 
         meta['country_b'] in slopes and meta['country_b'] in intercepts)
        for meta in metadata
    ])
    
    results = {
        'accuracy': accuracy,
        'precision': dict(zip(['A', 'B', 'ambiguous'], precision)),
        'recall': dict(zip(['A', 'B', 'ambiguous'], recall)),
        'f1': dict(zip(['A', 'B', 'ambiguous'], f1)),
        'support': dict(zip(['A', 'B', 'ambiguous'], support)),
        'confusion_matrix': cm,
        'coverage': coverage,
        'true_label_distribution': get_label_distribution(y_true_labels),
        'pred_label_distribution': get_label_distribution(y_pred_labels)
    }
    
    return results


def evaluate_all_methods(
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
    Evaluate prediction methods based on the methods list.
    Now includes discrete label evaluation.
    
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
        baseline_results = evaluate_baseline(y, N_diff, threshold)
        
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
            exchange_results = evaluate_exchange_rate_method(y, metadata, exchange_rates, N_frac, threshold)
            
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
            log_utility_results = evaluate_log_utility_method(
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
        pipeline = create_mlp_pipeline(
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
            label_encoder=label_encoder
        )
    
    # Decision Tree classifier with 5-fold CV on discrete labels
    if "decision_tree" in methods:
        print(f"Evaluating Decision Tree classifier with 5-fold CV on discrete labels (max_depth={max_depth})...")
        pipeline = create_decision_tree_pipeline(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        evaluate_classifier_with_cv(
            name="Decision Tree",
            pipeline=pipeline,
            X=X,
            y_labels=y_labels,
            label_order=['A', 'B', 'ambiguous'],
            cv_folds=5,
            random_state=random_state,
            y_is_encoded=False,
            label_encoder=None
        )


def print_data_summary(X: np.ndarray, y: np.ndarray, y_labels: np.ndarray, features: List[str], metadata: List[Dict[str, str]], threshold: float = 0.1) -> None:
    """Print summary information about the loaded data, including label distribution."""
    print(f"Loaded {len(X)} comparisons")
    print(f"Feature shape: {X.shape}")
    print(f"Features: {features}")
    print(f"Probability range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Threshold (t): {threshold}")
    print(f"Label distribution: {get_label_distribution(y_labels)}")
    print(f"Sample metadata: {metadata[0] if metadata else 'None'}")


def evaluate_equal_n_subset(
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
        
        evaluate_all_methods(
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


def train_and_analyze_mlp_criteria(
    X: np.ndarray,
    y_labels: np.ndarray,
    features: List[str],
    hidden_layer_sizes: Tuple[int, ...] = (5,),
    max_iter: int = 500,
    random_state: int = 42,
    alpha: float = 0.01
) -> None:
    """Train MLP classifier on full dataset and print learned criteria analysis."""
    print(f"\n{'='*60}")
    print("Training MLP classifier on full dataset and analyzing learned criteria")
    print(f"{'='*60}\n")
    
    # Encode string labels to numeric values for MLPClassifier
    label_encoder = LabelEncoder()
    y_labels_encoded = label_encoder.fit_transform(y_labels)
    
    # Create pipeline with scaling and MLP classifier
    pipeline = create_mlp_pipeline(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        alpha=alpha
    )
    
    # Train on full dataset with encoded discrete labels
    pipeline.fit(X, y_labels_encoded)
    
    # Extract the MLP from the pipeline
    mlp = pipeline.named_steps['mlp']
    
    # Get weights: coefs_[0] is input->hidden, coefs_[1] is hidden->output
    # For classifier: coefs_[0] is (n_features, n_hidden), coefs_[1] is (n_hidden, n_classes)
    input_to_hidden_weights = mlp.coefs_[0]  # Shape: (n_features, n_hidden)
    hidden_to_output_weights = mlp.coefs_[1]  # Shape: (n_hidden, n_classes)
    n_hidden = input_to_hidden_weights.shape[1]
    n_features = len(features)
    n_classes = len(mlp.classes_)
    
    # Decode class indices back to string labels
    class_labels = label_encoder.inverse_transform(mlp.classes_)
    
    print(f"Number of hidden neurons: {n_hidden}")
    print(f"Number of input features: {n_features}")
    print(f"Number of classes: {n_classes}")
    print(f"Classes: {class_labels}\n")
    
    # Analyze each hidden neuron as a learned criterion
    print("Learned Criteria (Hidden Neurons):")
    print("-" * 60)
    
    for neuron_idx in range(n_hidden):
        print(f"\nCriterion {neuron_idx + 1}:")
        print("  Output weights to classes:")
        for class_idx, class_name in enumerate(class_labels):
            print(f"    {class_name}: {hidden_to_output_weights[neuron_idx, class_idx]:.6f}")
        
        # Get input weights for this neuron
        input_weights = input_to_hidden_weights[:, neuron_idx]
        
        # Create list of (feature_name, weight, abs_weight) tuples
        feature_weight_pairs = [
            (features[i], input_weights[i], abs(input_weights[i]))
            for i in range(n_features)
        ]
        
        # Sort by absolute weight magnitude (descending)
        feature_weight_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print("  Input feature weights (ordered by magnitude):")
        for feature_name, weight, abs_weight in feature_weight_pairs:
            print(f"    {feature_name:30s}: {weight:10.6f}")
    
    print(f"\n{'='*60}")


def train_and_analyze_decision_tree(
    X: np.ndarray,
    y_labels: np.ndarray,
    features: List[str],
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> None:
    """Train Decision Tree classifier on full dataset and print learned feature importances and structure."""
    print(f"\n{'='*60}")
    print("Training Decision Tree classifier on full dataset and analyzing learned structure")
    print(f"{'='*60}\n")
    
    # Create pipeline with scaling and Decision Tree classifier
    pipeline = create_decision_tree_pipeline(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    # Train on full dataset with discrete labels
    pipeline.fit(X, y_labels)
    
    # Extract the Decision Tree from the pipeline
    dt = pipeline.named_steps['dt']
    
    # Get feature importances
    feature_importances = dt.feature_importances_
    n_features = len(features)
    
    # Print tree structure information
    print("Decision Tree Structure:")
    print(f"  Number of nodes: {dt.tree_.node_count}")
    print(f"  Tree depth: {dt.get_depth()}")
    print(f"  Number of leaves: {dt.tree_.n_leaves}")
    print(f"  Number of features: {n_features}")
    print(f"  Classes: {dt.classes_}")
    print()
    
    # Create list of (feature_name, importance, abs_importance) tuples
    feature_importance_pairs = [
        (features[i], feature_importances[i], abs(feature_importances[i]))
        for i in range(n_features)
    ]
    
    # Sort by absolute importance magnitude (descending)
    feature_importance_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("Feature Importances (ordered by magnitude):")
    print("-" * 60)
    for feature_name, importance, abs_importance in feature_importance_pairs:
        print(f"  {feature_name:30s}: {importance:10.6f}")
    
    print()
    print("Decision Rules (leaf nodes show predicted class labels):")
    print("-" * 60)
    # Note: export_text uses feature indices, but we want feature names
    # For classifiers, leaf nodes will show class labels instead of continuous values
    tree_rules = export_text(dt, feature_names=features, max_depth=max_depth, decimals=3)
    print(tree_rules)
    
    print(f"\n{'='*60}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate MLP models for decision modeling with discrete label mapping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name (used to construct CSV and JSONL file paths)"
    )
    
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=3,
        dest="hidden_dim",
        help="Number of hidden neurons in the MLP"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="L2 regularization parameter (higher = more sparsity/regularization)"
    )
    
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        dest="max_iter",
        help="Maximum number of iterations for MLP training"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        dest="random_state",
        help="Random state for reproducibility"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        dest="max_depth",
        help="Maximum depth for Decision Tree"
    )
    
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        dest="csv_path",
        help="Path to CSV file with country comparisons (default: {model_name}-country_vs_country.csv)"
    )
    
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default=None,
        dest="jsonl_path",
        help="Path to JSONL file with country features (default: country_features_{model_name}.jsonl)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        dest="threshold",
        help="Threshold parameter t for mapping scores to labels: 'A' if score >= 0.5+t, 'B' if score <= 0.5-t, 'ambiguous' otherwise"
    )
    
    parser.add_argument(
        "--no-include-ambiguous",
        action="store_false",
        default=True,
        dest="include_ambiguous",
        help="Exclude samples with 'ambiguous' label from training and evaluation (default: include ambiguous samples)"
    )
    
    parser.add_argument(
        "--log-utility-scale",
        type=float,
        default=0.28,
        dest="log_utility_scale",
        help="Scaling factor for log utility sigmoid/normal method (higher = sharper transition for sigmoid, std dev for normal)"
    )
    
    parser.add_argument(
        "--log-utility-method",
        type=str,
        default="normal",
        choices=["normal", "sigmoid"],
        dest="log_utility_method",
        help="Method for converting utility differences to probabilities: 'normal' (probit/Thurstonian) or 'sigmoid' (logistic)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Methods to include in evaluation
    methods = ["baseline", "exchange_rates", "log_utility", "mlp", "decision_tree"]
    #methods = ["baseline", "exchange_rates", "log_utility"]
    
    # Set default file paths if not provided
    csv_path = args.csv_path or f"{args.model_name}-country_vs_country.csv"
    jsonl_path = args.jsonl_path or f"country_features_{args.model_name}.jsonl"
    
    # Prepare MLP hyperparameters
    hidden_layer_sizes = (args.hidden_dim,)
    alpha = args.alpha
    
    # Load and preprocess data (now includes label mapping)
    X, y, y_labels, features, metadata, N_a, N_b = preprocess_decision_data(csv_path, jsonl_path, args.threshold)
    
    # Filter out ambiguous samples if requested
    if not args.include_ambiguous:
        non_ambiguous_mask = y_labels != 'ambiguous'
        X = X[non_ambiguous_mask]
        y = y[non_ambiguous_mask]
        y_labels = y_labels[non_ambiguous_mask]
        metadata = [metadata[i] for i in range(len(metadata)) if non_ambiguous_mask[i]]
        N_a = N_a[non_ambiguous_mask]
        N_b = N_b[non_ambiguous_mask]
        print(f"Filtered out {np.sum(~non_ambiguous_mask)} ambiguous samples. Remaining: {len(X)} samples.")
    
    # Print data summary (now includes label distribution)
    print_data_summary(X, y, y_labels, features, metadata, args.threshold)
    
    # Extract N_diff and N_frac (last two columns of X)
    N_diff = X[:, -2]
    N_frac = X[:, -1]
    
    # Evaluate on all data
    evaluate_all_methods(
        X, y, y_labels, metadata, N_a, N_b, N_diff, N_frac, args.model_name, methods, args.threshold, "all data",
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=args.max_iter,
        random_state=args.random_state,
        alpha=alpha,
        max_depth=args.max_depth,
        log_utility_scale=args.log_utility_scale,
        log_utility_method=args.log_utility_method
    )
    
    # Evaluate on N_a == N_b subset
    evaluate_equal_n_subset(
        X, y, y_labels, metadata, N_a, N_b, N_diff, N_frac, args.model_name, methods, args.threshold,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=args.max_iter,
        random_state=args.random_state,
        alpha=alpha,
        max_depth=args.max_depth,
        log_utility_scale=args.log_utility_scale,
        log_utility_method=args.log_utility_method
    )
    
    # Train MLP and analyze learned criteria (only if mlp is in methods)
    if "mlp" in methods:
        train_and_analyze_mlp_criteria(X, y_labels, features,
                                       hidden_layer_sizes=hidden_layer_sizes,
                                       max_iter=args.max_iter,
                                       random_state=args.random_state,
                                       alpha=alpha)
    
    # Train Decision Tree and analyze learned structure (only if decision_tree is in methods)
    if "decision_tree" in methods:
        train_and_analyze_decision_tree(X, y_labels, features,
                                        max_depth=args.max_depth,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        random_state=args.random_state)


if __name__ == "__main__":
    main()

