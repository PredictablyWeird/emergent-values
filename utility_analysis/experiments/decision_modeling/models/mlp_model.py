"""MLP model class with fit() and evaluate() methods for both regression and classification."""

import numpy as np
from typing import Tuple, Literal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .base import BaseModel
from .mlp import create_mlp_pipeline, create_mlp_classifier_pipeline


class MLPModel(BaseModel):
    """MLP model with fit() and evaluate() interface supporting both regression and classification."""
    
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (5,),
        max_iter: int = 500,
        random_state: int = 42,
        alpha: float = 0.01,
        task: Literal["regression", "classification"] = "regression"
    ):
        """
        Initialize MLP model.
        
        Args:
            hidden_layer_sizes: Tuple of hidden layer sizes
            max_iter: Maximum number of iterations
            random_state: Random state for reproducibility
            alpha: L2 regularization parameter (higher = more sparsity/regularization)
            task: "regression" or "classification"
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.task = task
        self.pipeline: Pipeline | None = None
        self.label_encoder: LabelEncoder | None = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'MLPModel':
        """
        Train the MLP model on training data.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target values (continuous for regression, discrete labels for classification)
            **kwargs: Ignored (kept for API consistency)
            
        Returns:
            self (for method chaining)
        """
        if self.task == "classification":
            self.pipeline = create_mlp_classifier_pipeline(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                random_state=self.random_state,
                alpha=self.alpha
            )
            # Encode labels for classification
            self.label_encoder = LabelEncoder()
            y_train_enc = self.label_encoder.fit_transform(y_train)
            self.pipeline.fit(X_train, y_train_enc)
        else:
            self.pipeline = create_mlp_pipeline(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                random_state=self.random_state,
                alpha=self.alpha
            )
            self.pipeline.fit(X_train, y_train)
        return self

