"""Unified classifier evaluation using shared CV splits."""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import confusion_matrix

from models.base import BaseModel
from data.preprocessing_discrete import get_label_distribution


def evaluate_classifier_with_cv(
    model: BaseModel,
    X: np.ndarray,
    y_labels: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    label_order: List[str] = ['A', 'B', 'ambiguous'],
    metadata: List[Dict[str, Any]] = None,
    N_a: np.ndarray = None,
    N_b: np.ndarray = None,
    N_frac: np.ndarray = None,
    y_probs: np.ndarray = None,
    **model_specific_kwargs
) -> Dict[str, Any]:
    """
    Evaluate a classifier model using shared CV splits.
    
    Args:
        model: Model instance (must be a BaseModel subclass with task="classification")
        X: Full feature matrix
        y_labels: Full target labels array
        cv_splits: List of (train_idx, test_idx) tuples from create_cv_splits()
        label_order: Ordered list of label names
        metadata: Optional list of metadata dictionaries (for exchange_rates/log_utility models)
        N_a: Optional array of N_a values (for exchange_rates/log_utility models)
        N_b: Optional array of N_b values (for exchange_rates/log_utility models)
        N_frac: Optional array of N_b / N_a ratios (for exchange_rates model)
        y_probs: Optional array of probability values (for models that need probabilities for training)
        **model_specific_kwargs: Additional kwargs passed to fit() and evaluate()
        
    Returns:
        Dictionary with aggregated CV results including mean and std of metrics
    """
    test_accuracies = []
    test_precisions = {label: [] for label in label_order}
    test_recalls = {label: [] for label in label_order}
    test_f1s = {label: [] for label in label_order}
    test_supports = {label: [] for label in label_order}
    
    train_accuracies = []
    train_precisions = {label: [] for label in label_order}
    train_recalls = {label: [] for label in label_order}
    train_f1s = {label: [] for label in label_order}
    train_supports = {label: [] for label in label_order}
    
    oof_pred_labels = np.empty_like(y_labels, dtype=object)
    
    for train_idx, test_idx in cv_splits:
        # Split data
        X_train, y_train_labels = X[train_idx], y_labels[train_idx]
        X_test, y_test_labels = X[test_idx], y_labels[test_idx]
        
        # For models that need probabilities for training (exchange_rates, log_utility),
        # use y_probs if provided, otherwise use y_train_labels
        if y_probs is not None:
            y_train = y_probs[train_idx]  # Use probabilities for training
        else:
            y_train = y_train_labels  # Use labels (for models that can train on labels)
        
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
        eval_kwargs['y_train'] = y_train_labels  # Use training labels for train metrics
        eval_kwargs['label_order'] = label_order
        
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
        
        # Evaluate model (use y_test_labels for classification evaluation)
        fold_results = model.evaluate(X_test, y_test_labels, **eval_kwargs)
        
        # Collect metrics
        test_accuracies.append(fold_results['test_accuracy'])
        for label in label_order:
            test_precisions[label].append(fold_results['test_precision'][label])
            test_recalls[label].append(fold_results['test_recall'][label])
            test_f1s[label].append(fold_results['test_f1'][label])
            test_supports[label].append(fold_results['test_support'][label])
        
        # Store out-of-fold predictions
        # Extract required arguments for predict() based on model type
        predict_kwargs = {}
        if metadata is not None:
            metadata_test = [metadata[i] for i in test_idx]
            predict_kwargs['metadata'] = metadata_test
        if N_a is not None:
            predict_kwargs['N_a'] = N_a[test_idx]
        if N_b is not None:
            predict_kwargs['N_b'] = N_b[test_idx]
        if N_frac is not None:
            predict_kwargs['N_frac'] = N_frac[test_idx]
        
        y_pred_test = model.predict(X_test, **predict_kwargs)
        
        # Convert to labels if needed (some models return probabilities, others return labels)
        threshold = model_specific_kwargs.get('threshold', 0.1)
        if len(y_pred_test) > 0 and isinstance(y_pred_test[0], (float, np.floating)):
            # Predictions are probabilities, convert to labels
            from data.preprocessing_discrete import map_scores_to_labels
            y_pred_test = map_scores_to_labels(y_pred_test, threshold)
        
        oof_pred_labels[test_idx] = y_pred_test
        
        # Collect train metrics if available
        if 'train_accuracy' in fold_results:
            train_accuracies.append(fold_results['train_accuracy'])
            for label in label_order:
                train_precisions[label].append(fold_results['train_precision'][label])
                train_recalls[label].append(fold_results['train_recall'][label])
                train_f1s[label].append(fold_results['train_f1'][label])
                train_supports[label].append(fold_results['train_support'][label])
    
    # Aggregate results
    results = {
        'cv_folds': len(cv_splits),
        'test_accuracy_mean': np.mean(test_accuracies),
        'test_accuracy_std': np.std(test_accuracies),
        'test_precision_mean': {label: np.mean(test_precisions[label]) for label in label_order},
        'test_precision_std': {label: np.std(test_precisions[label]) for label in label_order},
        'test_recall_mean': {label: np.mean(test_recalls[label]) for label in label_order},
        'test_recall_std': {label: np.std(test_recalls[label]) for label in label_order},
        'test_f1_mean': {label: np.mean(test_f1s[label]) for label in label_order},
        'test_f1_std': {label: np.std(test_f1s[label]) for label in label_order},
        'confusion_matrix_oof': confusion_matrix(y_labels, oof_pred_labels, labels=label_order),
        'true_label_distribution': get_label_distribution(y_labels),
        'pred_label_distribution_oof': get_label_distribution(oof_pred_labels),
    }
    
    if train_accuracies:
        results.update({
            'train_accuracy_mean': np.mean(train_accuracies),
            'train_accuracy_std': np.std(train_accuracies),
            'train_precision_mean': {label: np.mean(train_precisions[label]) for label in label_order},
            'train_precision_std': {label: np.std(train_precisions[label]) for label in label_order},
            'train_recall_mean': {label: np.mean(train_recalls[label]) for label in label_order},
            'train_recall_std': {label: np.std(train_recalls[label]) for label in label_order},
            'train_f1_mean': {label: np.mean(train_f1s[label]) for label in label_order},
            'train_f1_std': {label: np.std(train_f1s[label]) for label in label_order},
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

