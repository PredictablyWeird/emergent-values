"""Analysis functions for MLP models."""

import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder

from models import create_mlp_pipeline, create_mlp_classifier_pipeline


def train_and_analyze_mlp_criteria(
    X: np.ndarray,
    y: np.ndarray,
    features: List[str],
    hidden_layer_sizes: Tuple[int, ...] = (5,),
    max_iter: int = 500,
    random_state: int = 42,
    alpha: float = 0.01
) -> None:
    """Train MLP on full dataset and print learned criteria analysis."""
    print(f"\n{'='*60}")
    print("Training MLP on full dataset and analyzing learned criteria")
    print(f"{'='*60}\n")
    
    # Create pipeline with scaling and MLP (same configuration as in train_mlp_with_cv)
    pipeline = create_mlp_pipeline(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        alpha=alpha
    )
    
    # Train on full dataset
    pipeline.fit(X, y)
    
    # Extract the MLP from the pipeline
    mlp = pipeline.named_steps['mlp']
    
    # Get weights: coefs_[0] is input->hidden, coefs_[1] is hidden->output
    # Shape: coefs_[0] is (n_features, n_hidden), coefs_[1] is (n_hidden, 1)
    input_to_hidden_weights = mlp.coefs_[0]  # Shape: (n_features, n_hidden)
    hidden_to_output_weights = mlp.coefs_[1].flatten()  # Shape: (n_hidden,)
    n_hidden = len(hidden_to_output_weights)
    n_features = len(features)
    
    print(f"Number of hidden neurons: {n_hidden}")
    print(f"Number of input features: {n_features}\n")
    
    # Analyze each hidden neuron as a learned criterion
    print("Learned Criteria (Hidden Neurons):")
    print("-" * 60)
    
    for neuron_idx in range(n_hidden):
        print(f"\nCriterion {neuron_idx + 1}:")
        print(f"  Output weight: {hidden_to_output_weights[neuron_idx]:.6f}")
        
        # Get input weights for this neuron
        input_weights = input_to_hidden_weights[:, neuron_idx]
        
        # Create list of (feature_name, weight, abs_weight) tuples
        feature_weight_pairs = [
            (features[i], input_weights[i], abs(input_weights[i]))
            for i in range(n_features)
        ]
        
        # Sort by absolute weight magnitude (descending)
        feature_weight_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print("  Input feature weights (ordered by magnitude):")
        for feature_name, weight, abs_weight in feature_weight_pairs:
            print(f"    {feature_name:30s}: {weight:10.6f}")
    
    print(f"\n{'='*60}")


def train_and_analyze_mlp_criteria_discrete(
    X: np.ndarray,
    y_labels: np.ndarray,
    features: List[str],
    hidden_layer_sizes: Tuple[int, ...] = (5,),
    max_iter: int = 500,
    random_state: int = 42,
    alpha: float = 0.01
) -> None:
    """Train MLP classifier on full dataset and print learned criteria analysis."""
    print(f"\n{'='*60}")
    print("Training MLP classifier on full dataset and analyzing learned criteria")
    print(f"{'='*60}\n")
    
    # Encode string labels to numeric values for MLPClassifier
    label_encoder = LabelEncoder()
    y_labels_encoded = label_encoder.fit_transform(y_labels)
    
    # Create pipeline with scaling and MLP classifier
    pipeline = create_mlp_classifier_pipeline(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        alpha=alpha
    )
    
    # Train on full dataset with encoded discrete labels
    pipeline.fit(X, y_labels_encoded)
    
    # Extract the MLP from the pipeline
    mlp = pipeline.named_steps['mlp']
    
    # Get weights: coefs_[0] is input->hidden, coefs_[1] is hidden->output
    # For classifier: coefs_[0] is (n_features, n_hidden), coefs_[1] is (n_hidden, n_classes)
    input_to_hidden_weights = mlp.coefs_[0]  # Shape: (n_features, n_hidden)
    hidden_to_output_weights = mlp.coefs_[1]  # Shape: (n_hidden, n_classes)
    n_hidden = input_to_hidden_weights.shape[1]
    n_features = len(features)
    n_classes = len(mlp.classes_)
    
    # Decode class indices back to string labels
    class_labels = label_encoder.inverse_transform(mlp.classes_)
    
    print(f"Number of hidden neurons: {n_hidden}")
    print(f"Number of input features: {n_features}")
    print(f"Number of classes: {n_classes}")
    print(f"Classes: {class_labels}\n")
    
    # Analyze each hidden neuron as a learned criterion
    print("Learned Criteria (Hidden Neurons):")
    print("-" * 60)
    
    for neuron_idx in range(n_hidden):
        print(f"\nCriterion {neuron_idx + 1}:")
        print("  Output weights to classes:")
        for class_idx, class_name in enumerate(class_labels):
            print(f"    {class_name}: {hidden_to_output_weights[neuron_idx, class_idx]:.6f}")
        
        # Get input weights for this neuron
        input_weights = input_to_hidden_weights[:, neuron_idx]
        
        # Create list of (feature_name, weight, abs_weight) tuples
        feature_weight_pairs = [
            (features[i], input_weights[i], abs(input_weights[i]))
            for i in range(n_features)
        ]
        
        # Sort by absolute weight magnitude (descending)
        feature_weight_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print("  Input feature weights (ordered by magnitude):")
        for feature_name, weight, abs_weight in feature_weight_pairs:
            print(f"    {feature_name:30s}: {weight:10.6f}")
    
    print(f"\n{'='*60}")

