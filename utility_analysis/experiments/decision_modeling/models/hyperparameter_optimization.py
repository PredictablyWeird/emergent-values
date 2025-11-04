"""Hyperparameter optimization utilities for decision modeling models."""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import r2_score, accuracy_score


def optimize_scale_for_regression(
    model_fn,
    metadata_train: List[Dict[str, str]],
    N_a_train: np.ndarray,
    N_b_train: np.ndarray,
    y_train: np.ndarray,
    scale_range: Tuple[float, float] = (0.1, 2.0),
    num_points: int = 20,
    method: str = "normal"
) -> float:
    """
    Optimize scale parameter for regression by maximizing R² on training data.
    
    Args:
        model_fn: Function that takes (metadata, N_a, N_b, scale, method) and returns predictions
        metadata_train: Training metadata
        N_a_train: Training N_a values
        N_b_train: Training N_b values
        y_train: Training target probabilities
        scale_range: (min, max) range for scale values to search
        num_points: Number of scale values to try
        method: Method for predictions ("normal" or "sigmoid")
        
    Returns:
        Optimal scale value
    """
    scales = np.linspace(scale_range[0], scale_range[1], num_points)
    best_scale = scale_range[0]
    best_r2 = -np.inf
    
    for scale in scales:
        try:
            y_pred = model_fn(metadata_train, N_a_train, N_b_train, scale=scale, method=method)
            r2 = r2_score(y_train, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_scale = scale
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
    
    return best_scale


def optimize_scale_for_classification(
    model_fn,
    metadata_train: List[Dict[str, str]],
    N_a_train: np.ndarray,
    N_b_train: np.ndarray,
    y_train_labels: np.ndarray,
    scale_range: Tuple[float, float] = (0.1, 2.0),
    threshold: float = 0.1,
    num_points: int = 20,
    method: str = "normal"
) -> float:
    """
    Optimize scale parameter for classification by maximizing accuracy on training data.
    
    Args:
        model_fn: Function that takes (metadata, N_a, N_b, scale, method) and returns probability predictions
        metadata_train: Training metadata
        N_a_train: Training N_a values
        N_b_train: Training N_b values
        y_train_labels: Training target labels
        scale_range: (min, max) range for scale values to search
        threshold: Threshold for converting probabilities to labels
        num_points: Number of scale values to try
        method: Method for predictions ("normal" or "sigmoid")
        
    Returns:
        Optimal scale value
    """
    from data.preprocessing_discrete import map_scores_to_labels
    
    scales = np.linspace(scale_range[0], scale_range[1], num_points)
    best_scale = scale_range[0]
    best_accuracy = -1.0
    
    for scale in scales:
        try:
            y_pred_probs = model_fn(metadata_train, N_a_train, N_b_train, scale=scale, method=method)
            y_pred_labels = map_scores_to_labels(y_pred_probs, threshold)
            accuracy = accuracy_score(y_train_labels, y_pred_labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_scale = scale
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
    
    return best_scale


def optimize_tolerance_for_regression(
    predict_fn,
    metadata_train: List[Dict[str, str]],
    N_frac_train: np.ndarray,
    y_train: np.ndarray,
    exchange_rates: Dict,
    tolerance_range: Tuple[float, float] = (0.1, 0.5),
    num_points: int = 20
) -> float:
    """
    Optimize tolerance parameter for exchange rate predictions (regression).
    
    Args:
        predict_fn: Function that takes (metadata, exchange_rates, N_frac, tolerance) and returns predictions
        metadata_train: Training metadata
        N_frac_train: Training N_frac values
        y_train: Training target probabilities
        exchange_rates: Dictionary of exchange rates
        tolerance_range: (min, max) range for tolerance values to search
        num_points: Number of tolerance values to try
        
    Returns:
        Optimal tolerance value
    """
    tolerances = np.linspace(tolerance_range[0], tolerance_range[1], num_points)
    best_tolerance = tolerance_range[0]
    best_r2 = -np.inf
    
    for tolerance in tolerances:
        try:
            y_pred = predict_fn(metadata_train, exchange_rates, N_frac_train, tolerance=tolerance)
            r2 = r2_score(y_train, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_tolerance = tolerance
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
    
    return best_tolerance


def optimize_tolerance_for_classification(
    predict_fn,
    metadata_train: List[Dict[str, str]],
    N_frac_train: np.ndarray,
    y_train_labels: np.ndarray,
    exchange_rates: Dict,
    tolerance_range: Tuple[float, float] = (0.1, 0.5),
    threshold: float = 0.1,
    num_points: int = 20
) -> float:
    """
    Optimize tolerance parameter for exchange rate predictions (classification).
    
    Args:
        predict_fn: Function that takes (metadata, exchange_rates, N_frac, tolerance) and returns probability predictions
        metadata_train: Training metadata
        N_frac_train: Training N_frac values
        y_train_labels: Training target labels
        exchange_rates: Dictionary of exchange rates
        tolerance_range: (min, max) range for tolerance values to search
        threshold: Threshold for converting probabilities to labels
        num_points: Number of tolerance values to try
        
    Returns:
        Optimal tolerance value
    """
    from data.preprocessing_discrete import map_scores_to_labels
    
    tolerances = np.linspace(tolerance_range[0], tolerance_range[1], num_points)
    best_tolerance = tolerance_range[0]
    best_accuracy = -1.0
    
    for tolerance in tolerances:
        try:
            y_pred_probs = predict_fn(metadata_train, exchange_rates, N_frac_train, tolerance=tolerance)
            y_pred_labels = map_scores_to_labels(y_pred_probs, threshold)
            accuracy = accuracy_score(y_train_labels, y_pred_labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_tolerance = tolerance
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
    
    return best_tolerance

