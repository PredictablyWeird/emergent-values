import csv
import json
import numpy as np
from typing import Dict, List, Tuple
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def load_country_features(jsonl_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load country features from JSONL file.
    
    Returns:
        Dictionary mapping country names to their feature dictionaries.
    """
    country_features = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            country_name = record['country']
            features = record['features']
            country_features[country_name] = features
    return country_features


def load_decision_file(csv_path: str) -> List[Tuple[float, str, float, str, float]]:
    """
    Load decision file (CSV) with country comparisons.
    
    Returns:
        List of tuples: (N_a, country_a, N_b, country_b, probability)
    """
    decisions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # CSV format: N_a, country_a, N_b, country_b, probability
            if len(row) >= 5:
                N_a = float(row[0])
                country_a = row[1]
                N_b = float(row[2])
                country_b = row[3]
                probability = float(row[4])
                decisions.append((N_a, country_a, N_b, country_b, probability))
    return decisions


def create_feature_difference_vector(
    country_a_features: Dict[str, float],
    country_b_features: Dict[str, float],
    feature_names: List[str]
) -> np.ndarray:
    """
    Create feature difference vector: country_a_features - country_b_features.
    
    Args:
        country_a_features: Feature dictionary for country A
        country_b_features: Feature dictionary for country B
        feature_names: Ordered list of feature names
        
    Returns:
        Numpy array with feature differences
    """
    diff_vector = []
    for feature_name in feature_names:
        value_a = country_a_features.get(feature_name, 0.0)
        value_b = country_b_features.get(feature_name, 0.0)
        diff_vector.append(value_a - value_b)
    return np.array(diff_vector)


def preprocess_decision_data(
    csv_path: str,
    jsonl_path: str
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, str]]]:
    """
    Preprocess decision data from CSV and country features from JSONL.
    
    Args:
        csv_path: Path to CSV file with country comparisons
        jsonl_path: Path to JSONL file with country features
        
    Returns:
        Tuple of:
            - X: numpy array with input features (differences)
            - y: numpy array with probabilities
            - features: list of feature names
            - metadata: list of dictionaries with "country_a" and "country_b" keys
    """
    # Load country features
    country_features = load_country_features(jsonl_path)
    
    # Get feature names from the first country (assuming all have same features)
    if not country_features:
        raise ValueError("No country features found in JSONL file")
    
    first_country = next(iter(country_features))
    feature_names = sorted(country_features[first_country].keys())
    
    # Add N_diff feature name (will be appended at the end)
    feature_names = feature_names + ['N_diff']
    
    # Load decision data
    decisions = load_decision_file(csv_path)
    
    # Build feature matrices and metadata
    X_rows = []
    y_values = []
    metadata_list = []
    
    missing_countries = set()
    
    for N_a, country_a, N_b, country_b, probability in decisions:
        # Check if both countries have features
        if country_a not in country_features:
            missing_countries.add(country_a)
            continue
        if country_b not in country_features:
            missing_countries.add(country_b)
            continue
        
        # Get feature vectors
        features_a = country_features[country_a]
        features_b = country_features[country_b]
        
        # Create difference vector for country features
        diff_vector = create_feature_difference_vector(
            features_a, features_b, feature_names[:-1]  # Exclude N_diff from feature names here
        )
        
        # Calculate N_diff = N_a - N_b
        N_diff = N_a - N_b
        
        # Append N_diff to the feature vector
        feature_vector = np.append(diff_vector, N_diff)
        
        X_rows.append(feature_vector)
        y_values.append(probability)
        metadata_list.append({
            "country_a": country_a,
            "country_b": country_b
        })
    
    if missing_countries:
        print(f"Warning: Missing features for countries: {missing_countries}")
    
    # Convert to numpy arrays
    X = np.array(X_rows)
    y = np.array(y_values)
    
    return X, y, feature_names, metadata_list


def train_mlp_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    hidden_layer_sizes: Tuple[int, ...] = (5,),
    max_iter: int = 500,
    random_state: int = 42
) -> Dict:
    """
    Train MLP regressor using cross-validation to predict probabilities from features.
    
    Args:
        X: Input features array
        y: Target probabilities
        cv_folds: Number of cross-validation folds
        hidden_layer_sizes: Tuple of hidden layer sizes
        max_iter: Maximum number of iterations
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with cross-validation results including scores and metrics
    """
    # Create pipeline with scaling and MLP
    # Scaling is important for neural networks
    pipeline = Pipeline([
        #TODO Do we want scaling?
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            activation='tanh',
            validation_fraction=0.1
        )),
    ])
    
    # Perform cross-validation
    # Using multiple scoring metrics for regression
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
        'hidden_layer_sizes': hidden_layer_sizes,
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


def compute_baseline_predictions(N_diff: np.ndarray) -> np.ndarray:
    """
    Compute baseline predictions based on N_a vs N_b comparison.
    
    Args:
        N_diff: Array of N_a - N_b differences
        
    Returns:
        Array of baseline predictions:
        - 1.0 if N_a > N_b (N_diff > 0)
        - 0.0 if N_a < N_b (N_diff < 0)
        - 0.5 if N_a == N_b (N_diff == 0)
    """
    predictions = np.zeros_like(N_diff)
    predictions[N_diff > 0] = 1.0
    predictions[N_diff < 0] = 0.0
    predictions[N_diff == 0] = 0.5
    return predictions


def evaluate_baseline(y_true: np.ndarray, N_diff: np.ndarray) -> Dict:
    """
    Evaluate baseline predictor performance.
    
    Args:
        y_true: True probability values
        N_diff: Array of N_a - N_b differences
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = compute_baseline_predictions(N_diff)
    
    results = {
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred)
    }
    
    return results


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python decision_modeling_preprocessing.py <model_name>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    csv_path = f"{model_name}-country_vs_country.csv"
    jsonl_path = f"country_features_{model_name}.jsonl"
    
    X, y, features, metadata = preprocess_decision_data(csv_path, jsonl_path)
    
    print(f"Loaded {len(X)} comparisons")
    print(f"Feature shape: {X.shape}")
    print(f"Features: {features}")
    print(f"Probability range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Sample metadata: {metadata[0] if metadata else 'None'}")
    print()
    
    # Extract N_diff (last column of X)
    N_diff = X[:, -1]
    
    # Evaluate baseline
    print("Evaluating baseline predictor (predicts 1 if N_a > N_b, 0 if N_a < N_b, 0.5 if N_a == N_b)...")
    baseline_results = evaluate_baseline(y, N_diff)
    
    print("\nBaseline results:")
    print(f"  R² score: {baseline_results['r2']:.4f}")
    print(f"  MSE: {baseline_results['mse']:.4f}")
    print(f"  MAE: {baseline_results['mae']:.4f}")
    print()
    
    # Train MLP with cross-validation
    print("Training MLP regressor with cross-validation...")
    cv_results = train_mlp_with_cv(X, y, cv_folds=5)
    
    print(f"\nCross-validation results ({cv_results['cv_folds']}-fold CV):")
    print(f"Hidden layer sizes: {cv_results['hidden_layer_sizes']}")
    print()
    print("Test scores:")
    print(f"  R² score: {cv_results['test_r2_mean']:.4f} ± {cv_results['test_r2_std']:.4f}")
    print(f"  MSE: {cv_results['test_mse_mean']:.4f} ± {cv_results['test_mse_std']:.4f}")
    print(f"  MAE: {cv_results['test_mae_mean']:.4f} ± {cv_results['test_mae_std']:.4f}")
    print()
    print("Train scores:")
    print(f"  R² score: {cv_results['train_r2_mean']:.4f} ± {cv_results['train_r2_std']:.4f}")
    print(f"  MSE: {cv_results['train_mse_mean']:.4f} ± {cv_results['train_mse_std']:.4f}")
    print(f"  MAE: {cv_results['train_mae_mean']:.4f} ± {cv_results['train_mae_std']:.4f}")


if __name__ == "__main__":
    main()
