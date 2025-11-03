"""Log utility-based prediction methods."""

import os
import math
import pathlib
import numpy as np
from typing import Dict, List, Tuple

from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from data.preprocessing_discrete import map_scores_to_labels, get_label_distribution


def load_utility_curves(
    model_name: str,
    results_dir: str | None = None,
    category: str = "countries",
    measure: str = "terminal_illness"
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Load utility curve slopes and intercepts for countries from exchange rate results.
    
    Utility curves are of the form: utility = intercept + slope * ln(N)
    
    Args:
        model_name: Model name
        results_dir: Directory containing exchange rate results
        category: Category (e.g., 'countries')
        measure: Measure (e.g., 'terminal_illness')
        
    Returns:
        Tuple of (slopes, intercepts) dictionaries mapping country names to their slope/intercept values.
        Returns (empty dict, empty dict) if data not available.
    """
    import sys
    
    # Add path to import from create_exchange_rates_plots
    # Path: predictors/log_utility.py -> predictors/ -> decision_modeling/ -> experiments/ -> utility_analysis/
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(base_dir))
    
    try:
        from create_exchange_rates_plots import (
            load_thurstonian_results,
            fit_utility_curves
        )
    except ImportError:
        return {}, {}

    if results_dir is None:
        results_dir = base_dir / "experiments" / "exchange_rates" / "results"
    
    model_save_dir = os.path.join(results_dir, model_name)
    if not os.path.exists(model_save_dir):
        return {}, {}
    
    try:
        df, measure_vals = load_thurstonian_results(model_save_dir, category, measure)
        slopes, intercepts = fit_utility_curves(df, return_mse=False)
        return slopes, intercepts
    except (FileNotFoundError, ValueError):
        return {}, {}


def compute_log_utility_predictions(
    metadata: List[Dict[str, str]],
    slopes: Dict[str, float],
    intercepts: Dict[str, float],
    N_a: np.ndarray,
    N_b: np.ndarray,
    scale: float = 1.0,
    method: str = "normal"
) -> np.ndarray:
    """
    Compute probability predictions based on utility curves (slopes and intercepts) and quantities.
    
    Utility curve interpretation:
    - For country: utility = intercept + slope * ln(N)
    - For country_a with N_a: utility_a = intercept_a + slope_a * ln(N_a)
    - For country_b with N_b: utility_b = intercept_b + slope_b * ln(N_b)
    - Utility difference: utility_diff = utility_a - utility_b
    
    Args:
        metadata: List of dictionaries with 'country_a' and 'country_b' keys
        slopes: Dictionary mapping country names to their utility curve slopes
        intercepts: Dictionary mapping country names to their utility curve intercepts
        N_a: Array of N_a values
        N_b: Array of N_b values
        scale: Scaling factor for sigmoid (higher = sharper transition) or std dev for normal method
        method: "sigmoid" (logistic) or "normal" (probit/Thurstonian)
        
    Returns:
        Array of probability predictions
    """
    predictions = []
    
    for i, meta in enumerate(metadata):
        country_a = meta['country_a']
        country_b = meta['country_b']
        n_a = N_a[i]
        n_b = N_b[i]
        
        # Check if we have utility curve parameters for both countries
        if country_a in slopes and country_a in intercepts and country_b in slopes and country_b in intercepts:
            slope_a = slopes[country_a]
            intercept_a = intercepts[country_a]
            slope_b = slopes[country_b]
            intercept_b = intercepts[country_b]
            
            # Calculate utilities using: utility = intercept + slope * ln(N)
            # Handle edge cases where N <= 0
            if n_a > 0:
                utility_a = intercept_a + slope_a * math.log(n_a)
            else:
                utility_a = intercept_a  # ln(0) is undefined, use just intercept
            
            if n_b > 0:
                utility_b = intercept_b + slope_b * math.log(n_b)
            else:
                utility_b = intercept_b  # ln(0) is undefined, use just intercept
            
            utility_diff = utility_a - utility_b
            
            # Convert utility difference to probability
            if method == "sigmoid":
                # Logistic function
                probability = expit(scale * utility_diff)
            elif method == "normal":
                # Normal CDF (Thurstonian-style, assuming std dev = scale)
                probability = norm.cdf(utility_diff / scale) if scale > 0 else 0.5
            else:
                raise ValueError(f"Unknown method: {method}")
            
            predictions.append(np.clip(probability, 0.0, 1.0))
        else:
            # Utility curve parameters not available, predict 0.5 (indifference)
            predictions.append(0.5)
    
    return np.array(predictions)


def evaluate_log_utility_method(
    y_true: np.ndarray,
    metadata: List[Dict[str, str]],
    slopes: Dict[str, float],
    intercepts: Dict[str, float],
    N_a: np.ndarray,
    N_b: np.ndarray,
    scale: float = 1.0,
    method: str = "normal"
) -> Dict:
    """
    Evaluate log utility-based predictor performance.
    
    Args:
        y_true: True probability values
        metadata: List of dictionaries with 'country_a' and 'country_b' keys
        slopes: Dictionary mapping country names to their utility curve slopes
        intercepts: Dictionary mapping country names to their utility curve intercepts
        N_a: Array of N_a values
        N_b: Array of N_b values
        scale: Scaling factor for sigmoid or std dev for normal method
        method: "sigmoid" (logistic) or "normal" (probit/Thurstonian)
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = compute_log_utility_predictions(metadata, slopes, intercepts, N_a, N_b, scale, method)
    
    # Calculate coverage: fraction of comparisons where both countries have utility curves
    coverage = np.mean([
        (meta['country_a'] in slopes and meta['country_a'] in intercepts and 
         meta['country_b'] in slopes and meta['country_b'] in intercepts)
        for meta in metadata
    ])
    
    results = {
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'coverage': coverage
    }
    
    return results


def evaluate_log_utility_method_discrete(
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
        scale: Scaling factor for sigmoid or std dev for normal method
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

