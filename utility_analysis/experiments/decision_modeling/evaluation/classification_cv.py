"""Classification cross-validation utilities."""

import numpy as np
from typing import List, Dict
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from collections import Counter

from data.preprocessing_discrete import get_label_distribution


def evaluate_classifier_with_cv(
    name: str,
    pipeline: Pipeline,
    X: np.ndarray,
    y_labels: np.ndarray,
    label_order: List[str],
    cv_folds: int = 5,
    random_state: int = 42,
    y_is_encoded: bool = False,
    label_encoder: LabelEncoder = None,
    groups: np.ndarray = None
) -> None:
    """
    Evaluate a classifier using group-based K-Fold cross-validation.
    Groups partition the data by country_x so all rows sharing a country_x
    stay in the same fold. Prints mean±std metrics across folds and the
    out-of-fold confusion matrix.
    
    Args:
        name: Name of the classifier (for display)
        pipeline: Scikit-learn pipeline to evaluate
        X: Feature matrix
        y_labels: Array of discrete labels
        label_order: Ordered list of label names
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility
        y_is_encoded: Whether y_labels are already encoded as integers
        label_encoder: LabelEncoder to use if y_is_encoded is True
        groups: Array of group identifiers for GroupKFold
    """
    if groups is None:
        # Fallback to stratified if groups not provided
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X, y_labels)
    else:
        splitter = GroupKFold(n_splits=cv_folds)
        split_iter = splitter.split(X, y_labels, groups=groups)

    per_fold_metrics = []  # list of dicts with accuracy, precision/recall/f1 per-class (macro for summary)
    oof_pred_labels = np.empty_like(y_labels, dtype=object)

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        model = clone(pipeline)
        if y_is_encoded:
            # Encode using provided encoder
            y_train_enc = label_encoder.transform(y_labels[train_idx])
            model.fit(X[train_idx], y_train_enc)
            y_val_pred_enc = model.predict(X[val_idx])
            y_val_pred = label_encoder.inverse_transform(y_val_pred_enc)
        else:
            model.fit(X[train_idx], y_labels[train_idx])
            y_val_pred = model.predict(X[val_idx])

        y_true_val = y_labels[val_idx]
        oof_pred_labels[val_idx] = y_val_pred

        accuracy = accuracy_score(y_true_val, y_val_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_val, y_val_pred, labels=label_order, zero_division=0
        )

        per_fold_metrics.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        })

    # Aggregate per-fold metrics (means and stds)
    accuracies = np.array([m['accuracy'] for m in per_fold_metrics])
    precisions = np.stack([m['precision'] for m in per_fold_metrics], axis=0)
    recalls = np.stack([m['recall'] for m in per_fold_metrics], axis=0)
    f1s = np.stack([m['f1'] for m in per_fold_metrics], axis=0)

    print(f"\n{name} (5-fold CV) results:")
    print(f"  Accuracy: {accuracies.mean():.4f} ± {accuracies.std():.4f}")

    print("\nPer-class metrics (mean ± std across folds):")
    for i, label in enumerate(label_order):
        print(f"  {label}:")
        print(f"    Precision: {precisions[:, i].mean():.4f} ± {precisions[:, i].std():.4f}")
        print(f"    Recall:    {recalls[:, i].mean():.4f} ± {recalls[:, i].std():.4f}")
        print(f"    F1-score:  {f1s[:, i].mean():.4f} ± {f1s[:, i].std():.4f}")

    # Out-of-fold confusion matrix and distributions
    cm_oof = confusion_matrix(y_labels, oof_pred_labels, labels=label_order)
    print("\nOut-of-fold confusion matrix:")
    print(f"  {cm_oof}")
    print(f"\nTrue label distribution: {get_label_distribution(y_labels)}")
    print(f"Predicted label distribution (OOF): {get_label_distribution(oof_pred_labels)}")

