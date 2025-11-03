"""Data preprocessing utilities for decision modeling."""

import os
import pathlib
import numpy as np
from typing import Dict, List, Tuple

from data.loader import load_country_features, load_decision_file


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
    model_name: str,
    csv_path: str | None = None,
    jsonl_path: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, str]], np.ndarray, np.ndarray]:
    """
    Preprocess decision data from CSV and country features from JSONL.
    
    Args:
        model_name: Model name
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
    # Construct paths relative to utility_analysis directory
    # Path: data/preprocessing.py -> data/ -> decision_modeling/ -> experiments/ -> utility_analysis/
    if csv_path is None:
        base_dir = pathlib.Path(__file__).parent.parent.parent.parent
        csv_path = os.path.join(base_dir, f"{model_name}-country_vs_country.csv")
    if jsonl_path is None:
        base_dir = pathlib.Path(__file__).parent.parent.parent.parent
        jsonl_path = os.path.join(base_dir, f"country_features_{model_name}.jsonl")

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
    decisions = load_decision_file(model_name=model_name, csv_path=csv_path)
    
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

