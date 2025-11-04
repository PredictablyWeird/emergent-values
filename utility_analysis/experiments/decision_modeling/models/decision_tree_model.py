"""Decision Tree model class with fit() and evaluate() methods for both regression and classification."""

import numpy as np
from typing import Literal
from sklearn.pipeline import Pipeline

from .base import BaseModel
from .decision_tree import create_decision_tree_pipeline, create_decision_tree_classifier_pipeline


class DecisionTreeModel(BaseModel):
    """Decision Tree model with fit() and evaluate() interface supporting both regression and classification."""
    
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        task: Literal["regression", "classification"] = "regression"
    ):
        """
        Initialize Decision Tree model.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split a node
            min_samples_leaf: Minimum number of samples required at a leaf node
            random_state: Random state for reproducibility
            task: "regression" or "classification"
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.task = task
        self.pipeline: Pipeline | None = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'DecisionTreeModel':
        """
        Train the Decision Tree model on training data.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target values (continuous for regression, discrete labels for classification)
            **kwargs: Ignored (kept for API consistency)
            
        Returns:
            self (for method chaining)
        """
        if self.task == "classification":
            self.pipeline = create_decision_tree_classifier_pipeline(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        else:
            self.pipeline = create_decision_tree_pipeline(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        self.pipeline.fit(X_train, y_train)
        return self

