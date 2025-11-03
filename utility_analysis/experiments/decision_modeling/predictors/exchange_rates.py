"""Exchange rate-based prediction methods."""

import os
import math
import pathlib
import numpy as np
from typing import Dict, List, Tuple

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from data.preprocessing_discrete import map_scores_to_labels, get_label_distribution


def load_exchange_rates(
    model_name: str,
    results_dir: str | None = None,
    category: str = "countries",
    measure: str = "terminal_illness"
) -> Dict[Tuple[str, str], float]:
    """
    Load exchange rates for countries from exchange rate results.
    
    Args:
        model_name: Model name
        results_dir: Directory containing exchange rate results
        category: Category (e.g., 'countries')
        measure: Measure (e.g., 'terminal_illness')
        
    Returns:
        Dictionary mapping (country_a, country_b) tuples to exchange rate ratios.
        Returns empty dict if data not available.
    """
    import sys
    
    # Add path to import from create_exchange_rates_plots
    # Path: predictors/exchange_rates.py -> predictors/ -> decision_modeling/ -> experiments/ -> utility_analysis/
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(base_dir))
    
    try:
        from create_exchange_rates_plots import (
            load_thurstonian_results,
            fit_utility_curves,
            two_way_geometric_exchange_rate
        )
    except ImportError:
        return {}
    
    if results_dir is None:
        results_dir = base_dir / "experiments" / "exchange_rates" / "results"
    
    model_save_dir = os.path.join(results_dir, model_name)
    if not os.path.exists(model_save_dir):
        return {}
    
    try:
        df, measure_vals = load_thurstonian_results(model_save_dir, category, measure)
        slopes, intercepts = fit_utility_curves(df, return_mse=False)
    except (FileNotFoundError, ValueError):
        return {}
    
    # Build exchange rate dictionary
    exchange_rates = {}
    countries = list(slopes.keys())
    
    for country_a in countries:
        for country_b in countries:
            if country_a == country_b:
                continue
            exchange_rate = two_way_geometric_exchange_rate(
                country_a, country_b, measure_vals,
                slopes, intercepts,
                skip_if_negative_slope=False,
                allow_negative_slopes=True
            )
            if exchange_rate is not None and not math.isinf(exchange_rate):
                exchange_rates[(country_a, country_b)] = exchange_rate
    
    return exchange_rates


def compute_exchange_rate_predictions(
    metadata: List[Dict[str, str]],
    exchange_rates: Dict[Tuple[str, str], float],
    N_frac: np.ndarray,
    tolerance: float = 0.28,
) -> np.ndarray:
    """
    Compute predictions based on exchange rates and quantities.
    
    Exchange rate interpretation:
    - exchange_rate(country_a, country_b) = r means: r units of country_a = 1 unit of country_b
    - So N_a units of country_a = N_a / r units of country_b utility
    - N_b units of country_b = N_b units of country_b utility
    - Country A has higher total utility if: N_a / r > N_b
    - Which rearranges to: N_a / N_b > r, or N_b / N_a < 1 / r
    - Since N_frac = N_b / N_a, we compare: N_frac vs 1 / exchange_rate
    
    Args:
        metadata: List of dictionaries with 'country_a' and 'country_b' keys
        exchange_rates: Dictionary mapping (country_a, country_b) to exchange rate
        N_frac: Array of N_b / N_a ratios
        tolerance: Tolerance for determining when utilities are equal (default: 0.28)
        
    Returns:
        Array of predictions (probabilities: 1.0 for country A preferred, 0.0 for country B preferred, 0.5 for tie)
    """
    predictions = []
    
    for i, meta in enumerate(metadata):
        country_a = meta['country_a']
        country_b = meta['country_b']
        n_frac = N_frac[i]
        
        # Check if we have exchange rate for this pair
        if (country_a, country_b) in exchange_rates:
            exchange_rate = exchange_rates[(country_a, country_b)]
            # Convert exchange rate to per-unit utility ratio
            # exchange_rate > 1 means country_a needs more units, so utility_per_unit_A = 1/exchange_rate
            
            # Country A has higher total utility if: N_frac < 1 / exchange_rate
            if 1/n_frac > exchange_rate * (1 + tolerance):
                # Country A has higher total utility
                predictions.append(1.0)
            elif 1/n_frac < exchange_rate * (1 - tolerance):
                # Country B has higher total utility
                predictions.append(0.0)
            else:
                # Equal total utility
                predictions.append(0.5)
        else:
            # Exchange rate not available, predict 0.5
            predictions.append(0.5)
    
    return np.array(predictions)


def evaluate_exchange_rate_method(
    y_true: np.ndarray,
    metadata: List[Dict[str, str]],
    exchange_rates: Dict[Tuple[str, str], float],
    N_frac: np.ndarray
) -> Dict:
    """
    Evaluate exchange rate-based predictor performance.
    
    Args:
        y_true: True probability values
        metadata: List of dictionaries with 'country_a' and 'country_b' keys
        exchange_rates: Dictionary mapping (country_a, country_b) to exchange rate
        N_frac: Array of N_b / N_a ratios
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = compute_exchange_rate_predictions(metadata, exchange_rates, N_frac)
    
    results = {
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'coverage': np.mean([(meta['country_a'], meta['country_b']) in exchange_rates for meta in metadata])
    }
    
    return results


def evaluate_exchange_rate_method_discrete(
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

