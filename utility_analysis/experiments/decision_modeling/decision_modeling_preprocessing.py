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
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, str]], np.ndarray, np.ndarray]:
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
            - N_a: numpy array with N_a values
            - N_b: numpy array with N_b values
    """
    # Load country features
    country_features = load_country_features(jsonl_path)
    
    # Get feature names from the first country (assuming all have same features)
    if not country_features:
        raise ValueError("No country features found in JSONL file")
    
    first_country = next(iter(country_features))
    feature_names = sorted(country_features[first_country].keys())
    
    # Add N_diff and N_frac feature names (will be appended at the end)
    feature_names = feature_names + ['N_diff', 'N_frac']
    
    # Load decision data
    decisions = load_decision_file(csv_path)
    
    # Build feature matrices and metadata
    X_rows = []
    y_values = []
    metadata_list = []
    N_a_values = []
    N_b_values = []
    
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
            features_a, features_b, feature_names[:-2]  # Exclude N_diff and N_frac from feature names here
        )
        
        # Calculate N_diff = N_a - N_b and N_frac = N_b / N_a
        N_diff = N_a - N_b
        N_frac = N_b / N_a if N_a != 0 else 0.0
        
        # Append N_diff and N_frac to the feature vector
        feature_vector = np.append(diff_vector, [N_diff, N_frac])
        
        X_rows.append(feature_vector)
        y_values.append(probability)
        metadata_list.append({
            "country_a": country_a,
            "country_b": country_b
        })
        N_a_values.append(N_a)
        N_b_values.append(N_b)
    
    if missing_countries:
        print(f"Warning: Missing features for countries: {missing_countries}")
    
    # Convert to numpy arrays
    X = np.array(X_rows)
    y = np.array(y_values)
    N_a = np.array(N_a_values)
    N_b = np.array(N_b_values)
    
    return X, y, feature_names, metadata_list, N_a, N_b


def train_mlp_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    hidden_layer_sizes: Tuple[int, ...] = (5,),
    max_iter: int = 500,
    random_state: int = 42,
    alpha: float = 0.01
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
        alpha: L2 regularization parameter (higher = more sparsity/regularization)
        
    Returns:
        Dictionary with cross-validation results including scores and metrics
    """
    # Create pipeline with scaling and MLP
    # Scaling is important for neural networks
    # alpha parameter adds L2 regularization to encourage sparsity of weights
    pipeline = Pipeline([
        #TODO Do we want scaling?
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            activation='tanh',
            validation_fraction=0.1,
            alpha=alpha  # L2 regularization: higher values encourage sparser weights
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
        'alpha': alpha,
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


def load_exchange_rates(model_name: str, results_dir: str = "experiments/exchange_rates/results",
                       category: str = "countries", measure: str = "terminal_illness") -> Dict[Tuple[str, str], float]:
    """
    Load exchange rates for countries from exchange rate results.
    
    Args:
        model_name: Model name
        results_dir: Directory containing exchange rate results
        category: Category (e.g., 'countries')
        measure: Measure (e.g., 'terminal_illness')
        
    Returns:
        Dictionary mapping (country_a, country_b) tuples to exchange rate ratios.
        Returns empty dict if data not available.
    """
    import os
    import sys
    import math
    
    # Add path to import from create_exchange_rates_plots
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    try:
        from create_exchange_rates_plots import (
            load_thurstonian_results, fit_utility_curves, two_way_geometric_exchange_rate
        )
        from experiments.exchange_rates.evaluate_exchange_rates import N_values  # noqa: F401
    except ImportError:
        return {}
    
    model_save_dir = os.path.join(results_dir, model_name)
    if not os.path.exists(model_save_dir):
        return {}
    
    try:
        df, measure_vals = load_thurstonian_results(model_save_dir, category, measure)
        slopes, intercepts = fit_utility_curves(df, return_mse=False)
    except (FileNotFoundError, ValueError):
        return {}
    
    # Build exchange rate dictionary
    exchange_rates = {}
    countries = list(slopes.keys())
    
    for country_a in countries:
        for country_b in countries:
            if country_a == country_b:
                continue
            exchange_rate = two_way_geometric_exchange_rate(
                country_a, country_b, measure_vals,
                slopes, intercepts,
                skip_if_negative_slope=False,
                allow_negative_slopes=True
            )
            if exchange_rate is not None and not math.isinf(exchange_rate):
                exchange_rates[(country_a, country_b)] = exchange_rate
    
    return exchange_rates


def compute_exchange_rate_predictions(
    metadata: List[Dict[str, str]],
    exchange_rates: Dict[Tuple[str, str], float],
    N_frac: np.ndarray,
    tolerance: float = 0.2,
) -> np.ndarray:
    """
    Compute predictions based on exchange rates and quantities.
    
    Exchange rate interpretation:
    - exchange_rate(country_a, country_b) = r means: r units of country_a = 1 unit of country_b
    - So N_a units of country_a = N_a / r units of country_b utility
    - N_b units of country_b = N_b units of country_b utility
    - Country A has higher total utility if: N_a / r > N_b
    - Which rearranges to: N_a / N_b > r, or N_b / N_a < 1 / r
    - Since N_frac = N_b / N_a, we compare: N_frac vs 1 / exchange_rate
    
    Args:
        metadata: List of dictionaries with 'country_a' and 'country_b' keys
        exchange_rates: Dictionary mapping (country_a, country_b) to exchange rate
        N_frac: Array of N_b / N_a ratios
        
    Returns:
        Array of predictions (probabilities)
    """
    predictions = []
    
    for i, meta in enumerate(metadata):
        country_a = meta['country_a']
        country_b = meta['country_b']
        n_frac = N_frac[i]
        
        # Check if we have exchange rate for this pair
        if (country_a, country_b) in exchange_rates:
            exchange_rate = exchange_rates[(country_a, country_b)]
            # Convert exchange rate to per-unit utility ratio
            # exchange_rate > 1 means country_a needs more units, so utility_per_unit_A = 1/exchange_rate
            utility_ratio_threshold = 1.0 / exchange_rate
            
            # Country A has higher total utility if: N_frac < 1 / exchange_rate
            if n_frac < utility_ratio_threshold - tolerance:
                # Country A has higher total utility
                predictions.append(1.0)
            elif n_frac > utility_ratio_threshold + tolerance:
                # Country B has higher total utility
                predictions.append(0.0)
            else:
                # Equal total utility
                predictions.append(0.5)
        else:
            # Exchange rate not available, predict 0.5
            predictions.append(0.5)
    
    return np.array(predictions)


def evaluate_exchange_rate_method(
    y_true: np.ndarray,
    metadata: List[Dict[str, str]],
    exchange_rates: Dict[Tuple[str, str], float],
    N_frac: np.ndarray
) -> Dict:
    """
    Evaluate exchange rate-based predictor performance.
    
    Args:
        y_true: True probability values
        metadata: List of dictionaries with 'country_a' and 'country_b' keys
        exchange_rates: Dictionary mapping (country_a, country_b) to exchange rate
        N_frac: Array of N_b / N_a ratios
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = compute_exchange_rate_predictions(metadata, exchange_rates, N_frac)
    
    results = {
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'coverage': np.mean([(meta['country_a'], meta['country_b']) in exchange_rates for meta in metadata])
    }
    
    return results


def evaluate_all_methods(
    X: np.ndarray,
    y: np.ndarray,
    metadata: List[Dict[str, str]],
    N_diff: np.ndarray,
    N_frac: np.ndarray,
    model_name: str,
    subset_name: str = "all data"
) -> None:
    """
    Evaluate all prediction methods (baseline, exchange rate, MLP).
    
    Args:
        X: Feature matrix
        y: Target probabilities
        metadata: List of dictionaries with country information
        N_diff: Array of N_a - N_b differences
        N_frac: Array of N_b / N_a ratios
        model_name: Model name for loading exchange rates
        subset_name: Name of the subset being evaluated (for display)
    """
    print(f"\n{'='*60}")
    print(f"Evaluating methods on {subset_name}")
    print(f"{'='*60}")
    print(f"Number of comparisons: {len(X)}")
    print()
    
    # Evaluate baseline
    print("Evaluating baseline predictor (predicts 1 if N_a > N_b, 0 if N_a < N_b, 0.5 if N_a == N_b)...")
    baseline_results = evaluate_baseline(y, N_diff)
    
    print("\nBaseline results:")
    print(f"  R² score: {baseline_results['r2']:.4f}")
    print(f"  MSE: {baseline_results['mse']:.4f}")
    print(f"  MAE: {baseline_results['mae']:.4f}")
    print()
    
    # Try to load exchange rates and evaluate exchange rate method
    print("Attempting to load exchange rates...")
    exchange_rates = load_exchange_rates(model_name)
    
    if exchange_rates:
        print(f"Loaded {len(exchange_rates)} exchange rate pairs")
        print("Evaluating exchange rate-based predictor (using exchange rates and N_frac)...")
        exchange_results = evaluate_exchange_rate_method(y, metadata, exchange_rates, N_frac)
        
        print("\nExchange rate method results:")
        print(f"  R² score: {exchange_results['r2']:.4f}")
        print(f"  MSE: {exchange_results['mse']:.4f}")
        print(f"  MAE: {exchange_results['mae']:.4f}")
        print(f"  Coverage: {exchange_results['coverage']:.2%}")
        print()
    else:
        print("Exchange rate data not available - skipping exchange rate evaluation")
        print()
    
    # Train MLP with cross-validation
    # Using alpha=0.01 for L2 regularization to encourage sparsity of weights
    print("Training MLP regressor with cross-validation (L2 regularization, alpha=0.01)...")
    cv_results = train_mlp_with_cv(X, y, cv_folds=5, alpha=0.01)
    
    print(f"\nCross-validation results ({cv_results['cv_folds']}-fold CV):")
    print(f"Hidden layer sizes: {cv_results['hidden_layer_sizes']}")
    print(f"Alpha (L2 regularization): {cv_results['alpha']}")
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


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python decision_modeling_preprocessing.py <model_name>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    csv_path = f"{model_name}-country_vs_country.csv"
    jsonl_path = f"country_features_{model_name}.jsonl"
    
    X, y, features, metadata, N_a, N_b = preprocess_decision_data(csv_path, jsonl_path)
    
    print(f"Loaded {len(X)} comparisons")
    print(f"Feature shape: {X.shape}")
    print(f"Features: {features}")
    print(f"Probability range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Sample metadata: {metadata[0] if metadata else 'None'}")
    
    # Extract N_diff and N_frac (last two columns of X)
    N_diff = X[:, -2]
    N_frac = X[:, -1]
    
    # Evaluate on all data
    evaluate_all_methods(X, y, metadata, N_diff, N_frac, model_name, "all data")
    
    # Filter for N_a == N_b
    equal_n_mask = N_a == N_b
    if np.any(equal_n_mask):
        X_equal = X[equal_n_mask]
        y_equal = y[equal_n_mask]
        metadata_equal = [metadata[i] for i in range(len(metadata)) if equal_n_mask[i]]
        N_diff_equal = N_diff[equal_n_mask]
        N_frac_equal = N_frac[equal_n_mask]
        
        evaluate_all_methods(X_equal, y_equal, metadata_equal, N_diff_equal, N_frac_equal, 
                            model_name, "N_a == N_b subset")
    else:
        print("\nNo comparisons with N_a == N_b found.")


if __name__ == "__main__":
    main()
