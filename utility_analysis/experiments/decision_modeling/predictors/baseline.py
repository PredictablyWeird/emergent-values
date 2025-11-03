"""Baseline prediction methods."""

import numpy as np
from typing import Dict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from data.preprocessing_discrete import map_scores_to_labels, get_label_distribution


def compute_baseline_predictions(N_diff: np.ndarray) -> np.ndarray:
    """
    Compute baseline predictions based on N_a vs N_b comparison.
    
    Args:
        N_diff: Array of N_a - N_b differences
        
    Returns:
        Array of baseline predictions:
        - 1.0 if N_a > N_b (N_diff > 0)
        - 0.0 if N_a < N_b (N_diff < 0)
        - 0.5 if N_a == N_b (N_diff == 0)
    """
    predictions = np.zeros_like(N_diff)
    predictions[N_diff > 0] = 1.0
    predictions[N_diff < 0] = 0.0
    predictions[N_diff == 0] = 0.5
    return predictions


def evaluate_baseline(y_true: np.ndarray, N_diff: np.ndarray) -> Dict:
    """
    Evaluate baseline predictor performance.
    
    Args:
        y_true: True probability values
        N_diff: Array of N_a - N_b differences
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = compute_baseline_predictions(N_diff)
    
    results = {
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred)
    }
    
    return results


def evaluate_baseline_discrete(
    y_true: np.ndarray,
    N_diff: np.ndarray,
    threshold: float = 0.1
) -> Dict:
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

