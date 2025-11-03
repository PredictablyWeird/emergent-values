"""
Decision Modeling Preprocessing Script

This script trains and evaluates various models for predicting decision probabilities
based on country features and quantities. It supports multiple prediction methods:
- Baseline (simple N_a vs N_b comparison)
- Exchange rate-based predictions
- Log utility-based predictions
- MLP (Multi-Layer Perceptron) neural network
- Decision Tree regression
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data import preprocess_decision_data
from evaluation import (
    evaluate_all_methods,
    evaluate_equal_n_subset,
    print_data_summary,
)
from analysis import (
    train_and_analyze_mlp_criteria,
    train_and_analyze_decision_tree,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate models for decision modeling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
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
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Methods to include in evaluation
    methods = ["baseline", "exchange_rates", "log_utility", "mlp", "decision_tree"]
    
    # Prepare MLP hyperparameters
    hidden_layer_sizes = (args.hidden_dim,)
    alpha = args.alpha
    
    # Load and preprocess data
    X, y, features, metadata, N_a, N_b = preprocess_decision_data(
        model_name=args.model_name,
        csv_path=args.csv_path,
        jsonl_path=args.jsonl_path
    )
    
    # Print data summary
    print_data_summary(X, y, features, metadata)
    
    # Extract N_diff and N_frac (last two columns of X)
    N_diff = X[:, -2]
    N_frac = X[:, -1]
    
    # Evaluate on all data
    evaluate_all_methods(
        X, y, metadata, N_a, N_b, N_diff, N_frac,
        args.model_name, methods, "all data",
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=args.max_iter,
        random_state=args.random_state,
        alpha=alpha,
        max_depth=args.max_depth
    )
    
    # Evaluate on N_a == N_b subset
    evaluate_equal_n_subset(
        X, y, metadata, N_a, N_b, N_diff, N_frac,
        args.model_name, methods,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=args.max_iter,
        random_state=args.random_state,
        alpha=alpha,
        max_depth=args.max_depth
    )
    
    # Train MLP and analyze learned criteria (only if mlp is in methods)
    if "mlp" in methods:
        train_and_analyze_mlp_criteria(
            X, y, features,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=args.max_iter,
            random_state=args.random_state,
            alpha=alpha
        )
    
    # Train Decision Tree and analyze learned structure (only if decision_tree is in methods)
    if "decision_tree" in methods:
        train_and_analyze_decision_tree(
            X, y, features,
            max_depth=args.max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=args.random_state
        )


if __name__ == "__main__":
    main()
