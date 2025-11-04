"""Unified model evaluation using shared CV splits."""

import numpy as np
from typing import Dict, List, Tuple, Any

from models.base import BaseModel


def evaluate_model_with_cv(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    metadata: List[Dict[str, Any]] = None,
    N_a: np.ndarray = None,
    N_b: np.ndarray = None,
    N_frac: np.ndarray = None,
    **model_specific_kwargs
) -> Dict[str, float]:
    """
    Evaluate a model using shared CV splits.
    
    This function handles the common pattern of:
    1. Iterating over CV splits
    2. Training the model on each fold
    3. Evaluating on test set (and optionally train set)
    4. Aggregating metrics across folds
    
    Args:
        model: Model instance (must be a BaseModel subclass)
        X: Full feature matrix
        y: Full target array
        cv_splits: List of (train_idx, test_idx) tuples from create_cv_splits()
        metadata: Optional list of metadata dictionaries (for exchange_rates/log_utility models)
        N_a: Optional array of N_a values (for exchange_rates/log_utility models)
        N_b: Optional array of N_b values (for exchange_rates/log_utility models)
        N_frac: Optional array of N_b / N_a ratios (for exchange_rates model)
        **model_specific_kwargs: Additional kwargs passed to fit() and evaluate()
        
    Returns:
        Dictionary with aggregated CV results including mean and std of metrics
    """
    test_metrics = {'r2': [], 'mse': [], 'mae': []}
    train_metrics = {'r2': [], 'mse': [], 'mae': []}
    
    for train_idx, test_idx in cv_splits:
        # Split data
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Prepare fit kwargs
        fit_kwargs = model_specific_kwargs.copy()
        if metadata is not None:
            metadata_train = [metadata[i] for i in train_idx]
            fit_kwargs['metadata_train'] = metadata_train
        if N_a is not None:
            fit_kwargs['N_a_train'] = N_a[train_idx]
        if N_b is not None:
            fit_kwargs['N_b_train'] = N_b[train_idx]
        
        # Train model
        model.fit(X_train, y_train, **fit_kwargs)
        
        # Prepare evaluate kwargs
        eval_kwargs = model_specific_kwargs.copy()
        eval_kwargs['X_train'] = X_train
        eval_kwargs['y_train'] = y_train
        
        if metadata is not None:
            metadata_test = [metadata[i] for i in test_idx]
            eval_kwargs['metadata_test'] = metadata_test
            eval_kwargs['metadata_train'] = metadata_train
        if N_a is not None:
            eval_kwargs['N_a_test'] = N_a[test_idx]
            eval_kwargs['N_a_train'] = N_a[train_idx]
        if N_b is not None:
            eval_kwargs['N_b_test'] = N_b[test_idx]
            eval_kwargs['N_b_train'] = N_b[train_idx]
        if N_frac is not None:
            eval_kwargs['N_frac_test'] = N_frac[test_idx]
            if N_a is not None and N_b is not None:
                N_frac_train = N_b[train_idx] / N_a[train_idx]
                N_frac_train = np.where(N_a[train_idx] != 0, N_frac_train, 0.0)
                eval_kwargs['N_frac_train'] = N_frac_train
        
        # Evaluate model
        fold_results = model.evaluate(X_test, y_test, **eval_kwargs)
        
        # Collect metrics
        test_metrics['r2'].append(fold_results['test_r2'])
        test_metrics['mse'].append(fold_results['test_mse'])
        test_metrics['mae'].append(fold_results['test_mae'])
        
        if 'train_r2' in fold_results:
            train_metrics['r2'].append(fold_results['train_r2'])
            train_metrics['mse'].append(fold_results['train_mse'])
            train_metrics['mae'].append(fold_results['train_mae'])
    
    # Aggregate results
    results = {
        'cv_folds': len(cv_splits),
        'test_r2_mean': np.mean(test_metrics['r2']),
        'test_r2_std': np.std(test_metrics['r2']),
        'test_mse_mean': np.mean(test_metrics['mse']),
        'test_mse_std': np.std(test_metrics['mse']),
        'test_mae_mean': np.mean(test_metrics['mae']),
        'test_mae_std': np.std(test_metrics['mae']),
    }
    
    if train_metrics['r2']:
        results.update({
            'train_r2_mean': np.mean(train_metrics['r2']),
            'train_r2_std': np.std(train_metrics['r2']),
            'train_mse_mean': np.mean(train_metrics['mse']),
            'train_mse_std': np.std(train_metrics['mse']),
            'train_mae_mean': np.mean(train_metrics['mae']),
            'train_mae_std': np.std(train_metrics['mae']),
        })
    
    # Add model-specific hyperparameters from model instance
    if hasattr(model, 'hidden_layer_sizes'):
        results['hidden_layer_sizes'] = model.hidden_layer_sizes
        results['alpha'] = model.alpha
    if hasattr(model, 'max_depth'):
        results['max_depth'] = model.max_depth
        results['min_samples_split'] = model.min_samples_split
        results['min_samples_leaf'] = model.min_samples_leaf
    if hasattr(model, 'scale'):
        results['scale'] = model.scale
        results['method'] = model.method
        if hasattr(model, 'tolerance'):
            results['tolerance'] = model.tolerance
    
    return results

