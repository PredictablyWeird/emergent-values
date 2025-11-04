"""Decision Tree model creation and training."""

import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def create_decision_tree_pipeline(
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> Pipeline:
    """
    Create Decision Tree pipeline with scaling and Decision Tree regressor.
    
    Args:
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum number of samples required to split a node
        min_samples_leaf: Minimum number of samples required at a leaf node
        random_state: Random state for reproducibility
        
    Returns:
        Pipeline with scaler and Decision Tree regressor
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('dt', DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )),
    ])




def create_decision_tree_classifier_pipeline(
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> Pipeline:
    """
    Create Decision Tree pipeline with scaling and Decision Tree classifier.
    
    Args:
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum number of samples required to split a node
        min_samples_leaf: Minimum number of samples required at a leaf node
        random_state: Random state for reproducibility
        
    Returns:
        Pipeline with scaler and Decision Tree classifier
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('dt', DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )),
    ])

