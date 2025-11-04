"""Base model interface for decision modeling."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_recall_fscore_support
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


class BaseModel(ABC):
    """Base class for all decision modeling models."""
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'BaseModel':
        """
        Train the model on training data.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target values
            **kwargs: Additional training arguments (may include metadata, N_a, N_b, etc.)
            
        Returns:
            self (for method chaining)
        """
        pass
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        label_order: Optional[list] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        This method provides common evaluation logic for pipeline-based models.
        Subclasses can override this if they need custom evaluation logic.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target values (continuous for regression, discrete labels for classification)
            X_train: Optional training feature matrix (for computing train metrics)
            y_train: Optional training target values (for computing train metrics)
            label_order: Optional list of label names for classification (e.g., ['A', 'B', 'ambiguous'])
            **kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Check if this is a pipeline-based model
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            raise ValueError("Model must be fitted before evaluation. Call fit() first.")
        
        # Check if this model supports task-based evaluation
        if not hasattr(self, 'task'):
            raise NotImplementedError("Subclass must implement evaluate() or define 'task' attribute")
        
        if self.task == "classification":
            return self._evaluate_classification(
                X_test, y_test, X_train, y_train, label_order
            )
        else:
            return self._evaluate_regression(X_test, y_test, X_train, y_train)
    
    def _evaluate_classification(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: Optional[np.ndarray],
        y_train: Optional[np.ndarray],
        label_order: Optional[list]
    ) -> Dict[str, float]:
        """Evaluate classification task."""
        if label_order is None:
            label_order = ['A', 'B', 'ambiguous']
        
        # Get predictions
        y_pred_test = self._predict_for_evaluation(X_test)
        
        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test, recall_test, f1_test, support_test = precision_recall_fscore_support(
            y_test, y_pred_test, labels=label_order, zero_division=0
        )
        
        results = {
            'test_accuracy': accuracy_test,
            'test_precision': dict(zip(label_order, precision_test)),
            'test_recall': dict(zip(label_order, recall_test)),
            'test_f1': dict(zip(label_order, f1_test)),
            'test_support': dict(zip(label_order, support_test)),
        }
        
        if X_train is not None and y_train is not None:
            y_pred_train = self._predict_for_evaluation(X_train)
            
            accuracy_train = accuracy_score(y_train, y_pred_train)
            precision_train, recall_train, f1_train, support_train = precision_recall_fscore_support(
                y_train, y_pred_train, labels=label_order, zero_division=0
            )
            
            results.update({
                'train_accuracy': accuracy_train,
                'train_precision': dict(zip(label_order, precision_train)),
                'train_recall': dict(zip(label_order, recall_train)),
                'train_f1': dict(zip(label_order, f1_train)),
                'train_support': dict(zip(label_order, support_train)),
            })
        
        return results
    
    def _evaluate_regression(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: Optional[np.ndarray],
        y_train: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate regression task."""
        y_pred_test = self._predict_for_evaluation(X_test)
        
        results = {
            'test_r2': r2_score(y_test, y_pred_test),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
        }
        
        if X_train is not None and y_train is not None:
            y_pred_train = self._predict_for_evaluation(X_train)
            results.update({
                'train_r2': r2_score(y_train, y_pred_train),
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
            })
        
        return results
    
    def _predict_for_evaluation(self, X: np.ndarray) -> np.ndarray:
        """
        Helper method to get predictions for evaluation.
        Handles label encoding/decoding for classification models.
        """
        if self.task == "classification" and hasattr(self, 'label_encoder') and self.label_encoder is not None:
            y_pred_enc = self.pipeline.predict(X)
            return self.label_encoder.inverse_transform(y_pred_enc)
        else:
            return self.pipeline.predict(X)
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            **kwargs: Additional prediction arguments
            
        Returns:
            Array of predictions
        """
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            return self._predict_for_evaluation(X)
        raise NotImplementedError("Subclasses should implement predict() if not using pipeline-based evaluation")

