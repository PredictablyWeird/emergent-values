"""
Decision Modeling Preprocessing Script (Discrete Labels)

This script trains and evaluates various models for predicting discrete decision labels
(A, B, or ambiguous) based on country features and quantities. It supports multiple 
prediction methods:
- Baseline (simple N_a vs N_b comparison)
- Exchange rate-based predictions
- Log utility-based predictions
- MLP (Multi-Layer Perceptron) neural network classifier
- Decision Tree classifier

The discrete labels are created by mapping continuous probabilities to discrete classes
using a threshold parameter.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessing_discrete import preprocess_decision_data_discrete
from evaluation import (
    evaluate_all_methods_discrete,
    evaluate_equal_n_subset_discrete,
    print_data_summary_discrete,
)
from analysis.mlp_analysis import train_and_analyze_mlp_criteria_discrete
from analysis.decision_tree_analysis import train_and_analyze_decision_tree_discrete


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate models for decision modeling with discrete label mapping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-4o-mini",
        help="Model name (used to construct CSV and JSONL file paths)"
    )
    
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=3,
        dest="hidden_dim",
        help="Number of hidden neurons in the MLP"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="L2 regularization parameter (higher = more sparsity/regularization)"
    )
    
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        dest="max_iter",
        help="Maximum number of iterations for MLP training"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        dest="random_state",
        help="Random state for reproducibility"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        dest="max_depth",
        help="Maximum depth for Decision Tree"
    )
    
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        dest="csv_path",
        help="Path to CSV file with country comparisons (default: {model_name}-country_vs_country.csv)"
    )
    
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default=None,
        dest="jsonl_path",
        help="Path to JSONL file with country features (default: country_features_{model_name}.jsonl)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        dest="threshold",
        help="Threshold parameter t for mapping scores to labels: 'A' if score >= 0.5+t, 'B' if score <= 0.5-t, 'ambiguous' otherwise"
    )
    
    parser.add_argument(
        "--no-include-ambiguous",
        action="store_false",
        default=True,
        dest="include_ambiguous",
        help="Exclude samples with 'ambiguous' label from training and evaluation (default: include ambiguous samples)"
    )
    
    parser.add_argument(
        "--log-utility-scale",
        type=float,
        default=0.28,
        dest="log_utility_scale",
        help="Scaling factor for log utility sigmoid/normal method (higher = sharper transition for sigmoid, std dev for normal)"
    )
    
    parser.add_argument(
        "--log-utility-method",
        type=str,
        default="normal",
        choices=["normal", "sigmoid"],
        dest="log_utility_method",
        help="Method for converting utility differences to probabilities: 'normal' (probit/Thurstonian) or 'sigmoid' (logistic)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Methods to include in evaluation
    #methods = ["baseline", "exchange_rates_cv", "log_utility_cv", "mlp", "decision_tree"]
    methods = ["baseline", "exchange_rates_cv", "log_utility_cv", "thurstonian_cv", "decision_tree"]
    
    # Prepare MLP hyperparameters
    hidden_layer_sizes = (args.hidden_dim,)
    alpha = args.alpha
    
    # Load and preprocess data (now includes label mapping)
    X, y, y_labels, features, metadata, N_a, N_b = preprocess_decision_data_discrete(
        model_name=args.model_name, 
        csv_path=args.csv_path,
        jsonl_path=args.jsonl_path,
        threshold=args.threshold
    )
    
    # Filter out ambiguous samples if requested
    if not args.include_ambiguous:
        non_ambiguous_mask = y_labels != 'ambiguous'
        X = X[non_ambiguous_mask]
        y = y[non_ambiguous_mask]
        y_labels = y_labels[non_ambiguous_mask]
        metadata = [metadata[i] for i in range(len(metadata)) if non_ambiguous_mask[i]]
        N_a = N_a[non_ambiguous_mask]
        N_b = N_b[non_ambiguous_mask]
        print(f"Filtered out {np.sum(~non_ambiguous_mask)} ambiguous samples. Remaining: {len(X)} samples.")
    
    # Print data summary (now includes label distribution)
    print_data_summary_discrete(X, y, y_labels, features, metadata, args.threshold)
    
    # Extract N_diff and N_frac (last two columns of X)
    N_diff = X[:, -2]
    N_frac = X[:, -1]
    
    # Evaluate on all data
    evaluate_all_methods_discrete(
        X, y, y_labels, metadata, N_a, N_b, N_diff, N_frac,
        args.model_name, methods, args.threshold, "all data",
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=args.max_iter,
        random_state=args.random_state,
        alpha=alpha,
        max_depth=args.max_depth,
        log_utility_scale=args.log_utility_scale,
        log_utility_method=args.log_utility_method
    )
    
    # Evaluate on N_a == N_b subset
    evaluate_equal_n_subset_discrete(
        X, y, y_labels, metadata, N_a, N_b, N_diff, N_frac,
        args.model_name, methods, args.threshold,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=args.max_iter,
        random_state=args.random_state,
        alpha=alpha,
        max_depth=args.max_depth,
        log_utility_scale=args.log_utility_scale,
        log_utility_method=args.log_utility_method
    )
    
    # Train MLP and analyze learned criteria (only if mlp is in methods)
    if "mlp" in methods:
        train_and_analyze_mlp_criteria_discrete(
            X, y_labels, features,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=args.max_iter,
            random_state=args.random_state,
            alpha=alpha
        )
    
    # Train Decision Tree and analyze learned structure (only if decision_tree is in methods)
    if "decision_tree" in methods:
        train_and_analyze_decision_tree_discrete(
            X, y_labels, features,
            max_depth=args.max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=args.random_state
        )


if __name__ == "__main__":
    main()
