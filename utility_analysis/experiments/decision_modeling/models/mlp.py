"""MLP (Multi-Layer Perceptron) model creation and training."""

from typing import Tuple
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def create_mlp_pipeline(
    hidden_layer_sizes: Tuple[int, ...] = (5,),
    max_iter: int = 500,
    random_state: int = 42,
    alpha: float = 0.01
) -> Pipeline:
    """
    Create MLP pipeline with scaling and MLP regressor.
    
    Args:
        hidden_layer_sizes: Tuple of hidden layer sizes
        max_iter: Maximum number of iterations
        random_state: Random state for reproducibility
        alpha: L2 regularization parameter (higher = more sparsity/regularization)
        
    Returns:
        Pipeline with scaler and MLP regressor
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            activation='tanh',
            validation_fraction=0.1,
            alpha=alpha
        )),
    ])




def create_mlp_classifier_pipeline(
    hidden_layer_sizes: Tuple[int, ...] = (5,),
    max_iter: int = 500,
    random_state: int = 42,
    alpha: float = 0.01
) -> Pipeline:
    """
    Create MLP pipeline with scaling and MLP classifier.
    
    Args:
        hidden_layer_sizes: Tuple of hidden layer sizes
        max_iter: Maximum number of iterations
        random_state: Random state for reproducibility
        alpha: L2 regularization parameter (higher = more sparsity/regularization)
        
    Returns:
        Pipeline with scaler and MLP classifier
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            activation='tanh',
            validation_fraction=0.1,
            alpha=alpha
        )),
    ])

