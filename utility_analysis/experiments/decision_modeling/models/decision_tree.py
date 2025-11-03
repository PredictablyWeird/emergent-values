"""Decision Tree model creation and training."""

import numpy as np
from typing import Dict
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate


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


def train_decision_tree_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> Dict:
    """
    Train Decision Tree regressor using cross-validation to predict probabilities from features.
    
    Args:
        X: Input features array
        y: Target probabilities
        cv_folds: Number of cross-validation folds
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum number of samples required to split a node
        min_samples_leaf: Minimum number of samples required at a leaf node
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with cross-validation results including scores and metrics
    """
    pipeline = create_decision_tree_pipeline(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    # Perform cross-validation
    scoring = {
        'r2': 'r2',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'neg_mean_absolute_error': 'neg_mean_absolute_error'
    }
    
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv_folds,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Calculate mean and std for each metric
    results = {
        'cv_folds': cv_folds,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'test_r2_mean': np.mean(cv_results['test_r2']),
        'test_r2_std': np.std(cv_results['test_r2']),
        'test_mse_mean': -np.mean(cv_results['test_neg_mean_squared_error']),
        'test_mse_std': np.std(cv_results['test_neg_mean_squared_error']),
        'test_mae_mean': -np.mean(cv_results['test_neg_mean_absolute_error']),
        'test_mae_std': np.std(cv_results['test_neg_mean_absolute_error']),
        'train_r2_mean': np.mean(cv_results['train_r2']),
        'train_r2_std': np.std(cv_results['train_r2']),
        'train_mse_mean': -np.mean(cv_results['train_neg_mean_squared_error']),
        'train_mse_std': np.std(cv_results['train_neg_mean_squared_error']),
        'train_mae_mean': -np.mean(cv_results['train_neg_mean_absolute_error']),
        'train_mae_std': np.std(cv_results['train_neg_mean_absolute_error']),
    }
    
    return results


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

