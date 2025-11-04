"""Log utility model class with fit() and evaluate() methods."""

import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_recall_fscore_support
)

from .base import BaseModel
from .exchange_rates import fit_utility_curves_from_data
from predictors.log_utility import compute_log_utility_predictions
from data.preprocessing_discrete import map_scores_to_labels
from .hyperparameter_optimization import (
    optimize_scale_for_regression,
    optimize_scale_for_classification
)


class LogUtilityModel(BaseModel):
    """Log utility-based model with fit() and evaluate() interface."""
    
    def __init__(
        self,
        scale: float = 1.0,
        method: str = "normal",
        optimize_scale: bool = True
    ):
        """
        Initialize Log Utility model.
        
        Args:
            scale: Scaling factor for predictions (converts utility differences to probabilities).
                   Not used for fitting utility curves, only for predictions. Used as initial value
                   if optimize_scale=False, otherwise optimized during fit().
            method: "normal" (probit) or "sigmoid" (logistic) for converting utility differences to probabilities
            optimize_scale: If True, optimize scale on training data during fit() (optimizes prediction scale)
            
        Note:
            The scale parameter is used for predictions only (not for fitting utility curves).
            When optimize_scale=True, the scale is optimized after utility curves are fitted.
        """
        self.scale = scale
        self.method = method
        self.optimize_scale = optimize_scale
        self.slopes: Dict[str, float] | None = None
        self.intercepts: Dict[str, float] | None = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        metadata_train: List[Dict[str, str]],
        N_a_train: np.ndarray,
        N_b_train: np.ndarray,
        **kwargs
    ) -> 'LogUtilityModel':
        """
        Train the Log Utility model by fitting utility curves.
        
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
        
        # Optimize scale if requested (scale is used for predictions, not for fitting utility curves)
        if self.optimize_scale and self.slopes and self.intercepts:
            # Check if this is classification (y_train contains string labels)
            is_classification = len(y_train) > 0 and isinstance(y_train[0], (str, np.str_))
            
            # Create a prediction function that uses the fitted utility curves
            def predict_with_scale(metadata, N_a, N_b, scale, method):
                return compute_log_utility_predictions(
                    metadata, self.slopes, self.intercepts, N_a, N_b,
                    scale=scale, method=method
                )
            
            if is_classification:
                # Get threshold from kwargs if provided
                threshold = kwargs.get('threshold', 0.1)
                self.scale = optimize_scale_for_classification(
                    predict_with_scale,
                    metadata_train, N_a_train, N_b_train, y_train,
                    scale_range=(0.1, 2.0),
                    threshold=threshold,
                    method=self.method
                )
            else:
                self.scale = optimize_scale_for_regression(
                    predict_with_scale,
                    metadata_train, N_a_train, N_b_train, y_train,
                    scale_range=(0.1, 2.0),
                    method=self.method
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
        Evaluate the trained Log Utility model on test data.
        
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
        if not self.slopes or not self.intercepts:
            # If fitting failed, use default predictions
            y_pred_test = np.full(len(y_test), 0.5)
        else:
            y_pred_test = compute_log_utility_predictions(
                metadata_test, self.slopes, self.intercepts, N_a_test, N_b_test,
                scale=self.scale, method=self.method
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
                if not self.slopes or not self.intercepts:
                    y_pred_train = np.full(len(y_train), 0.5)
                else:
                    y_pred_train = compute_log_utility_predictions(
                        metadata_train, self.slopes, self.intercepts, N_a_train, N_b_train,
                        scale=self.scale, method=self.method
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
                if not self.slopes or not self.intercepts:
                    y_pred_train = np.full(len(y_train), 0.5)
                else:
                    y_pred_train = compute_log_utility_predictions(
                        metadata_train, self.slopes, self.intercepts, N_a_train, N_b_train,
                        scale=self.scale, method=self.method
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
            Array of predictions
        """
        if not self.slopes or not self.intercepts:
            return np.full(len(metadata), 0.5)
        
        return compute_log_utility_predictions(
            metadata, self.slopes, self.intercepts, N_a, N_b,
            scale=self.scale, method=self.method
        )

