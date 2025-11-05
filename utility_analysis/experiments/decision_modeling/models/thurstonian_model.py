"""Thurstonian model class that directly uses option utilities (mean and variance)."""

import numpy as np
import math
from typing import Dict, List, Tuple
from scipy.stats import norm
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_recall_fscore_support
)

from .base import BaseModel
from data.preprocessing_discrete import map_scores_to_labels


def fit_thurstonian_utilities_from_data(
    metadata: List[Dict[str, str]],
    N_a: np.ndarray,
    N_b: np.ndarray,
    y: np.ndarray,
) -> Dict[Tuple[str, float], Dict[str, float]]:
    """
    Fit Thurstonian model to get utilities (mean, variance) for each (country, N) option.
    
    Args:
        metadata: List of dictionaries with 'country_a' and 'country_b' keys
        N_a: Array of N_a values
        N_b: Array of N_b values
        y: Array of probabilities (probability that country A is preferred)
        
    Returns:
        Dictionary mapping (country, N) tuples to {'mean': float, 'variance': float}
    """
    import sys
    import pathlib
    # Add utility_analysis directory to path
    utility_analysis_dir = pathlib.Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(utility_analysis_dir))
    
    # Import Thurstonian model components
    from compute_utilities.compute_utilities import PreferenceGraph, PreferenceEdge
    
    # Create unique options for each (country, N) combination
    option_id_to_option = {}
    option_counter = 0
    
    for i, meta in enumerate(metadata):
        country_a = meta['country_a']
        country_b = meta['country_b']
        n_a = float(N_a[i])
        n_b = float(N_b[i])
        
        # Create option keys for (country, N) pairs
        option_a_key = (country_a, n_a)
        option_b_key = (country_b, n_b)
        
        # Assign IDs to options if not seen before
        if option_a_key not in option_id_to_option:
            option_id_to_option[option_a_key] = {
                'id': option_counter,
                'description': f"{country_a} with N={n_a}",
                'country': country_a,
                'N': n_a
            }
            option_counter += 1
        
        if option_b_key not in option_id_to_option:
            option_id_to_option[option_b_key] = {
                'id': option_counter,
                'description': f"{country_b} with N={n_b}",
                'country': country_b,
                'N': n_b
            }
            option_counter += 1
    
    # Create preference graph
    options_list = [opt for opt in option_id_to_option.values()]
    if len(options_list) < 2:
        return {}
    
    graph = PreferenceGraph(options=options_list, holdout_fraction=0.0, seed=42)
    
    # Add edges to graph
    for i, meta in enumerate(metadata):
        country_a = meta['country_a']
        country_b = meta['country_b']
        n_a = float(N_a[i])
        n_b = float(N_b[i])
        prob_a = float(y[i])
        
        option_a_key = (country_a, n_a)
        option_b_key = (country_b, n_b)
        
        option_A = option_id_to_option[option_a_key]
        option_B = option_id_to_option[option_b_key]
        
        # Clamp probability to avoid infinity
        prob_a = np.clip(prob_a, 1e-6, 1 - 1e-6)
        
        edge = PreferenceEdge(
            option_A=option_A,
            option_B=option_B,
            probability_A=prob_a,
            aux_data={}
        )
        
        edge_index = (option_A['id'], option_B['id'])
        graph.edges[edge_index] = edge
        if edge_index in graph.training_edges_pool:
            graph.training_edges_pool.remove(edge_index)
        graph.training_edges.add(edge_index)
    
    # Fit Thurstonian model
    try:
        from compute_utilities.utility_models.thurstonian.utils import fit_thurstonian_model
        utilities_dict, _, _ = fit_thurstonian_model(
            graph=graph,
            num_epochs=1000,
            learning_rate=0.01
        )
    except Exception as e:
        print(f"Error fitting Thurstonian model: {e}")
        return {}
    
    # Convert from option_id -> utilities to (country, N) -> utilities
    option_utilities = {}
    for option_key, option_data in option_id_to_option.items():
        option_id = option_data['id']
        if option_id in utilities_dict:
            option_utilities[option_key] = utilities_dict[option_id]
    
    return option_utilities


def compute_thurstonian_predictions(
    metadata: List[Dict[str, str]],
    N_a: np.ndarray,
    N_b: np.ndarray,
    option_utilities: Dict[Tuple[str, float], Dict[str, float]]
) -> np.ndarray:
    """
    Compute predictions using Thurstonian model utilities.
    
    For each pair (A, B), compute:
    - mu_A, var_A = utilities for (country_a, N_a)
    - mu_B, var_B = utilities for (country_b, N_b)
    - P(A preferred) = CDF((mu_A - mu_B) / sqrt(var_A + var_B))
    
    Args:
        metadata: List of dictionaries with 'country_a' and 'country_b' keys
        N_a: Array of N_a values
        N_b: Array of N_b values
        option_utilities: Dictionary mapping (country, N) to {'mean': float, 'variance': float}
        
    Returns:
        Array of probability predictions
    """
    predictions = []
    
    for i, meta in enumerate(metadata):
        country_a = meta['country_a']
        country_b = meta['country_b']
        n_a = float(N_a[i])
        n_b = float(N_b[i])
        
        option_a_key = (country_a, n_a)
        option_b_key = (country_b, n_b)
        
        # Check if we have utilities for both options
        if option_a_key in option_utilities and option_b_key in option_utilities:
            util_a = option_utilities[option_a_key]
            util_b = option_utilities[option_b_key]
            
            mu_a = util_a['mean']
            var_a = util_a['variance']
            mu_b = util_b['mean']
            var_b = util_b['variance']
            
            # Compute probability: P(A preferred) = CDF((mu_A - mu_B) / sqrt(var_A + var_B))
            variance = var_a + var_b + 1e-5  # Add small epsilon for numerical stability
            delta = mu_a - mu_b
            z = delta / np.sqrt(variance)
            
            # Use normal CDF
            probability = norm.cdf(z)
            predictions.append(np.clip(probability, 0.0, 1.0))
        else:
            # Utilities not available, predict 0.5 (indifference)
            predictions.append(0.5)
    
    return np.array(predictions)


class ThurstonianModel(BaseModel):
    """Thurstonian model that directly uses option utilities (mean and variance)."""
    
    def __init__(self):
        """
        Initialize Thurstonian model.
        """
        self.option_utilities: Dict[Tuple[str, float], Dict[str, float]] | None = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        metadata_train: List[Dict[str, str]],
        N_a_train: np.ndarray,
        N_b_train: np.ndarray,
        **kwargs
    ) -> 'ThurstonianModel':
        """
        Train the Thurstonian model by fitting utilities for each (country, N) option.
        
        Args:
            X_train: Training feature matrix (not used, kept for API consistency)
            y_train: Training target values (probabilities)
            metadata_train: List of dictionaries with country information
            N_a_train: Array of N_a values for training data
            N_b_train: Array of N_b values for training data
            **kwargs: Ignored (kept for API consistency)
            
        Returns:
            self (for method chaining)
        """
        self.option_utilities = fit_thurstonian_utilities_from_data(
            metadata_train, N_a_train, N_b_train, y_train
        )
        
        return self
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metadata_test: List[Dict[str, str]],
        N_a_test: np.ndarray,
        N_b_test: np.ndarray,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None,
        metadata_train: List[Dict[str, str]] = None,
        N_a_train: np.ndarray = None,
        N_b_train: np.ndarray = None,
        threshold: float = 0.1,
        label_order: List[str] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate the trained Thurstonian model on test data.
        
        Supports both regression (when y_test contains probabilities) and classification
        (when y_test contains discrete labels like 'A', 'B', 'ambiguous').
        
        Args:
            X_test: Test feature matrix (not used, kept for API consistency)
            y_test: Test target values (probabilities or discrete labels)
            metadata_test: List of dictionaries with country information for test data
            N_a_test: Array of N_a values for test data
            N_b_test: Array of N_b values for test data
            X_train: Optional training feature matrix (for computing train metrics)
            y_train: Optional training target values (for computing train metrics)
            metadata_train: Optional training metadata (for computing train metrics)
            N_a_train: Optional training N_a values (for computing train metrics)
            N_b_train: Optional training N_b values (for computing train metrics)
            threshold: Threshold for converting probabilities to labels (for classification)
            label_order: List of label names for classification (e.g., ['A', 'B', 'ambiguous'])
            **kwargs: Ignored (kept for API consistency)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.option_utilities is None:
            # If fitting failed, use default predictions
            y_pred_test = np.full(len(y_test), 0.5)
        else:
            y_pred_test = compute_thurstonian_predictions(
                metadata_test, N_a_test, N_b_test, self.option_utilities
            )
        
        # Check if this is classification (y_test contains string labels)
        is_classification = len(y_test) > 0 and isinstance(y_test[0], (str, np.str_))
        
        if is_classification:
            # Classification: convert predictions to labels and use classification metrics
            if label_order is None:
                label_order = ['A', 'B', 'ambiguous']
            
            y_pred_test_labels = map_scores_to_labels(y_pred_test, threshold)
            
            accuracy_test = accuracy_score(y_test, y_pred_test_labels)
            precision_test, recall_test, f1_test, support_test = precision_recall_fscore_support(
                y_test, y_pred_test_labels, labels=label_order, zero_division=0
            )
            
            results = {
                'test_accuracy': accuracy_test,
                'test_precision': dict(zip(label_order, precision_test)),
                'test_recall': dict(zip(label_order, recall_test)),
                'test_f1': dict(zip(label_order, f1_test)),
                'test_support': dict(zip(label_order, support_test)),
            }
            
            if (X_train is not None and y_train is not None and 
                metadata_train is not None and N_a_train is not None and N_b_train is not None):
                y_pred_train = compute_thurstonian_predictions(
                    metadata_train, N_a_train, N_b_train, self.option_utilities
                )
                
                y_pred_train_labels = map_scores_to_labels(y_pred_train, threshold)
                
                accuracy_train = accuracy_score(y_train, y_pred_train_labels)
                precision_train, recall_train, f1_train, support_train = precision_recall_fscore_support(
                    y_train, y_pred_train_labels, labels=label_order, zero_division=0
                )
                
                results.update({
                    'train_accuracy': accuracy_train,
                    'train_precision': dict(zip(label_order, precision_train)),
                    'train_recall': dict(zip(label_order, recall_train)),
                    'train_f1': dict(zip(label_order, f1_train)),
                    'train_support': dict(zip(label_order, support_train)),
                })
        else:
            # Regression: use regression metrics
            results = {
                'test_r2': r2_score(y_test, y_pred_test),
                'test_mse': mean_squared_error(y_test, y_pred_test),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
            }
            
            if (X_train is not None and y_train is not None and 
                metadata_train is not None and N_a_train is not None and N_b_train is not None):
                y_pred_train = compute_thurstonian_predictions(
                    metadata_train, N_a_train, N_b_train, self.option_utilities
                )
                
                results.update({
                    'train_r2': r2_score(y_train, y_pred_train),
                    'train_mse': mean_squared_error(y_train, y_pred_train),
                    'train_mae': mean_absolute_error(y_train, y_pred_train),
                })
        
        return results
    
    def predict(
        self,
        X: np.ndarray,
        metadata: List[Dict[str, str]],
        N_a: np.ndarray,
        N_b: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix (not used, kept for API consistency)
            metadata: List of dictionaries with country information
            N_a: Array of N_a values
            N_b: Array of N_b values
            **kwargs: Ignored (kept for API consistency)
            
        Returns:
            Array of probability predictions
        """
        if self.option_utilities is None:
            return np.full(len(metadata), 0.5)
        
        return compute_thurstonian_predictions(
            metadata, N_a, N_b, self.option_utilities
        )

