"""Analysis functions for Decision Tree models."""

from typing import List
from sklearn.tree import export_text

from models import create_decision_tree_pipeline, create_decision_tree_classifier_pipeline


def train_and_analyze_decision_tree(
    X,
    y,
    features: List[str],
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> None:
    """Train Decision Tree on full dataset and print learned feature importances and structure."""
    print(f"\n{'='*60}")
    print("Training Decision Tree on full dataset and analyzing learned structure")
    print(f"{'='*60}\n")
    
    # Create pipeline with scaling and Decision Tree (same configuration as in train_decision_tree_with_cv)
    pipeline = create_decision_tree_pipeline(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    # Train on full dataset
    pipeline.fit(X, y)
    
    # Extract the Decision Tree from the pipeline
    dt = pipeline.named_steps['dt']
    
    # Get feature importances
    feature_importances = dt.feature_importances_
    n_features = len(features)
    
    # Print tree structure information
    print("Decision Tree Structure:")
    print(f"  Number of nodes: {dt.tree_.node_count}")
    print(f"  Tree depth: {dt.get_depth()}")
    print(f"  Number of leaves: {dt.tree_.n_leaves}")
    print(f"  Number of features: {n_features}")
    print()
    
    # Create list of (feature_name, importance, abs_importance) tuples
    feature_importance_pairs = [
        (features[i], feature_importances[i], abs(feature_importances[i]))
        for i in range(n_features)
    ]
    
    # Sort by absolute importance magnitude (descending)
    feature_importance_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("Feature Importances (ordered by magnitude):")
    print("-" * 60)
    for feature_name, importance, abs_importance in feature_importance_pairs:
        print(f"  {feature_name:30s}: {importance:10.6f}")
    
    print()
    print("Decision Rules:")
    print("-" * 60)
    # Note: export_text uses feature indices, but we want feature names
    # We'll use the feature names we have
    tree_rules = export_text(dt, feature_names=features, max_depth=max_depth, decimals=3)
    print(tree_rules)
    
    print(f"\n{'='*60}")


def train_and_analyze_decision_tree_discrete(
    X,
    y_labels,
    features: List[str],
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> None:
    """Train Decision Tree classifier on full dataset and print learned feature importances and structure."""
    print(f"\n{'='*60}")
    print("Training Decision Tree classifier on full dataset and analyzing learned structure")
    print(f"{'='*60}\n")
    
    # Create pipeline with scaling and Decision Tree classifier
    pipeline = create_decision_tree_classifier_pipeline(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    # Train on full dataset with discrete labels
    pipeline.fit(X, y_labels)
    
    # Extract the Decision Tree from the pipeline
    dt = pipeline.named_steps['dt']
    
    # Get feature importances
    feature_importances = dt.feature_importances_
    n_features = len(features)
    
    # Print tree structure information
    print("Decision Tree Structure:")
    print(f"  Number of nodes: {dt.tree_.node_count}")
    print(f"  Tree depth: {dt.get_depth()}")
    print(f"  Number of leaves: {dt.tree_.n_leaves}")
    print(f"  Number of features: {n_features}")
    print(f"  Classes: {dt.classes_}")
    print()
    
    # Create list of (feature_name, importance, abs_importance) tuples
    feature_importance_pairs = [
        (features[i], feature_importances[i], abs(feature_importances[i]))
        for i in range(n_features)
    ]
    
    # Sort by absolute importance magnitude (descending)
    feature_importance_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("Feature Importances (ordered by magnitude):")
    print("-" * 60)
    for feature_name, importance, abs_importance in feature_importance_pairs:
        print(f"  {feature_name:30s}: {importance:10.6f}")
    
    print()
    print("Decision Rules (leaf nodes show predicted class labels):")
    print("-" * 60)
    # Note: export_text uses feature indices, but we want feature names
    # For classifiers, leaf nodes will show class labels instead of continuous values
    tree_rules = export_text(dt, feature_names=features, max_depth=max_depth, decimals=3)
    print(tree_rules)
    
    print(f"\n{'='*60}")

