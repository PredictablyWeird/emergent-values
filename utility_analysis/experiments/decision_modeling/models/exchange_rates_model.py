"""Exchange rate model class with fit() and evaluate() methods."""

import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_recall_fscore_support
)

from .base import BaseModel
from .exchange_rates import (
    fit_utility_curves_from_data,
    estimate_exchange_rates_from_utility_curves,
)
from predictors.exchange_rates import compute_exchange_rate_predictions
from data.preprocessing_discrete import map_scores_to_labels
from .hyperparameter_optimization import (
    optimize_tolerance_for_regression,
    optimize_tolerance_for_classification
)


class ExchangeRatesModel(BaseModel):
    """Exchange rate-based model with fit() and evaluate() interface."""
    
    def __init__(
        self,
        scale: float = 1.0,
        method: str = "normal",
        tolerance: float = 0.28,
        optimize_tolerance: bool = False
    ):
        """
        Initialize Exchange Rates model.
        
        Args:
            scale: Scaling factor for utility curve fitting (kept for API compatibility)
            method: "normal" (probit) or "sigmoid" (logistic) - only "normal" is supported
            tolerance: Tolerance for exchange rate predictions (used if optimize_tolerance=False)
            optimize_tolerance: If True, optimize tolerance on training data during fit()
        """
        self.scale = scale
        self.method = method
        self.tolerance = tolerance
        self.optimize_tolerance = optimize_tolerance
        self.slopes: Dict[str, float] | None = None
        self.intercepts: Dict[str, float] | None = None
        self.exchange_rates: Dict | None = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        metadata_train: List[Dict[str, str]],
        N_a_train: np.ndarray,
        N_b_train: np.ndarray,
        **kwargs
    ) -> 'ExchangeRatesModel':
        """
        Train the Exchange Rates model by fitting utility curves.
        
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
        self.slopes, self.intercepts = fit_utility_curves_from_data(
            metadata_train, N_a_train, N_b_train, y_train,
            scale=self.scale, method=self.method
        )
        
        if self.slopes and self.intercepts:
            self.exchange_rates = estimate_exchange_rates_from_utility_curves(
                self.slopes, self.intercepts
            )
        else:
            self.exchange_rates = {}
        
        # Optimize tolerance if requested
        if self.optimize_tolerance and self.exchange_rates:
            # Check if this is classification (y_train contains string labels)
            is_classification = len(y_train) > 0 and isinstance(y_train[0], (str, np.str_))
            
            N_frac_train = N_b_train / N_a_train
            N_frac_train = np.where(N_a_train != 0, N_frac_train, 0.0)
            
            if is_classification:
                # Get threshold from kwargs if provided
                threshold = kwargs.get('threshold', 0.1)
                self.tolerance = optimize_tolerance_for_classification(
                    compute_exchange_rate_predictions,
                    metadata_train, N_frac_train, y_train,
                    self.exchange_rates,
                    tolerance_range=(0.1, 0.5),
                    threshold=threshold
                )
            else:
                self.tolerance = optimize_tolerance_for_regression(
                    compute_exchange_rate_predictions,
                    metadata_train, N_frac_train, y_train,
                    self.exchange_rates,
                    tolerance_range=(0.1, 0.5)
                )
        
        return self
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metadata_test: List[Dict[str, str]],
        N_frac_test: np.ndarray,
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
        Evaluate the trained Exchange Rates model on test data.
        
        Supports both regression (when y_test contains probabilities) and classification
        (when y_test contains discrete labels like 'A', 'B', 'ambiguous').
        
        Args:
            X_test: Test feature matrix (not used, kept for API consistency)
            y_test: Test target values (probabilities or discrete labels)
            metadata_test: List of dictionaries with country information for test data
            N_frac_test: Array of N_b / N_a ratios for test data
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
        if self.slopes is None or self.intercepts is None:
            # If fitting failed, use default predictions
            y_pred_test = np.full(len(y_test), 0.5)
        elif not self.exchange_rates:
            y_pred_test = np.full(len(y_test), 0.5)
        else:
            y_pred_test = compute_exchange_rate_predictions(
                metadata_test, self.exchange_rates, N_frac_test, tolerance=self.tolerance
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
                N_frac_train = N_b_train / N_a_train
                N_frac_train = np.where(N_a_train != 0, N_frac_train, 0.0)
                
                if not self.slopes or not self.intercepts or not self.exchange_rates:
                    y_pred_train = np.full(len(y_train), 0.5)
                else:
                    y_pred_train = compute_exchange_rate_predictions(
                        metadata_train, self.exchange_rates, N_frac_train, tolerance=self.tolerance
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
                N_frac_train = N_b_train / N_a_train
                N_frac_train = np.where(N_a_train != 0, N_frac_train, 0.0)
                
                if not self.slopes or not self.intercepts or not self.exchange_rates:
                    y_pred_train = np.full(len(y_train), 0.5)
                else:
                    y_pred_train = compute_exchange_rate_predictions(
                        metadata_train, self.exchange_rates, N_frac_train, tolerance=self.tolerance
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
        N_frac: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix (not used, kept for API consistency)
            metadata: List of dictionaries with country information
            N_frac: Array of N_b / N_a ratios
            **kwargs: Ignored (kept for API consistency)
            
        Returns:
            Array of predictions
        """
        if self.slopes is None or self.intercepts is None or not self.exchange_rates:
            return np.full(len(metadata), 0.5)
        
        return compute_exchange_rate_predictions(
            metadata, self.exchange_rates, N_frac, tolerance=self.tolerance
        )

