"""Cross-validation utilities for decision modeling."""

import numpy as np
from typing import Iterator, Tuple, List
from sklearn.model_selection import KFold, GroupKFold


def create_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
    groups: np.ndarray = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create cross-validation splits globally.
    
    Args:
        X: Feature matrix
        y: Target values
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility
        groups: Optional group identifiers for GroupKFold
        
    Returns:
        List of (train_idx, test_idx) tuples for each fold
    """
    if groups is not None:
        splitter = GroupKFold(n_splits=cv_folds)
        split_iter = splitter.split(X, y, groups=groups)
    else:
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X, y)
    
    return list(split_iter)

