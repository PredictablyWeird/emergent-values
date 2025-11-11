#!/usr/bin/env python3
"""
Generate exchange rate plots for stated preferences experiments.

This script adapts stated preferences results to work with the existing
create_exchange_rates_plots.py plotting functions.
"""

import os
import json
import sys
import argparse
import numpy as np
import pandas as pd

# Get the absolute path to utility_analysis directory
current_dir = os.path.dirname(os.path.abspath(__file__))
utility_analysis_dir = os.path.abspath(os.path.join(current_dir, "../.."))
exchange_rates_dir = os.path.join(utility_analysis_dir, "experiments/exchange_rates")

# Add directories to path
if utility_analysis_dir not in sys.path:
    sys.path.insert(0, utility_analysis_dir)
if exchange_rates_dir not in sys.path:
    sys.path.insert(0, exchange_rates_dir)

# Now import
from create_exchange_rates_plots import (
    plot_appendix_multi_model_average,
    fit_utility_curves
)


def load_stated_preferences_results(results_dir, prompt_config, model):
    """
    Load results from stated preferences format.
    
    Args:
        results_dir: Base results directory (e.g., 'results_stated/')
        prompt_config: Prompt config name
        model: Model name
    
    Returns:
        DataFrame with columns: option_id, description, utility_mean, utility_variance, N, X
    """
    results_path = os.path.join(results_dir, prompt_config, model)
    
    if not os.path.isdir(results_path):
        raise FileNotFoundError(f"'{results_path}' doesn't exist.")
    
    # Find results file
    json_files = [f for f in os.listdir(results_path)
                  if f.startswith("results_") and f.endswith(".json")]
    if not json_files:
        raise FileNotFoundError(f"No 'results_*.json' found in '{results_path}'.")
    
    json_file_path = os.path.join(results_path, json_files[0])
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    options = data.get("options", [])
    utilities = data.get("utilities", {})
    
    if not options or not utilities:
        raise ValueError(f"No 'options' or 'utilities' in {json_file_path}.")
    
    rows = []
    for opt in options:
        opt_id = opt["id"]
        desc = opt["description"]
        str_opt_id = str(opt_id)
        
        if str_opt_id not in utilities:
            continue
        
        util_mean = utilities[str_opt_id]["mean"]
        util_var = utilities[str_opt_id]["variance"]
        
        # Get N and X directly from option
        N_val = opt.get("N")
        X_val = opt.get("X")
        
        rows.append({
            "option_id": opt_id,
            "description": desc,
            "utility_mean": util_mean,
            "utility_variance": util_var,
            "N": N_val,
            "X": X_val
        })
    
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["N", "X"])
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df = df[df["N"] > 0]
    df["lnN"] = np.log(df["N"])
    
    # Get unique X values and N values from the data
    X_values = sorted(df["X"].unique())
    N_values_list = sorted(df["N"].unique())
    
    return df, X_values, N_values_list


def create_plots_for_stated_preferences(
    results_dir,
    prompt_config,
    model,
    canonical_X=None,
    include_Xs=None,
    plot_scale='log',
    value_interpretation=None,
    output_dir=None
):
    """
    Create plots for stated preferences results.
    
    Args:
        results_dir: Base results directory
        prompt_config: Prompt config name
        model: Model name
        canonical_X: Reference X value for ratios (default: first X)
        include_Xs: List of X values to include (default: all)
        plot_scale: 'log' or 'linear'
        value_interpretation: 'positive' or 'negative'
        output_dir: Directory to save plots (default: same as results)
    
    Returns:
        Dictionary of figures
    """
    # Load data
    df, X_values, N_values_list = load_stated_preferences_results(
        results_dir, prompt_config, model
    )
    
    # Set defaults
    if canonical_X is None:
        canonical_X = X_values[0]
    
    if include_Xs is None:
        include_Xs = X_values
    else:
        # Filter to only include_Xs that exist in data
        include_Xs = [x for x in include_Xs if x in X_values]
    
    if output_dir is None:
        output_dir = os.path.join(results_dir, prompt_config, model)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a temporary directory structure that the plotting code expects
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Create the expected directory structure: temp_dir/model/dummy/dummy/
        model_dir = os.path.join(temp_dir, model, 'dummy', 'dummy')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save our data in a format the loader expects
        temp_data = {
            "options": [],
            "utilities": {}
        }
        
        for _, row in df.iterrows():
            opt_id = row["option_id"]
            temp_data["options"].append({
                "id": opt_id,
                "description": row["description"]
            })
            temp_data["utilities"][str(opt_id)] = {
                "mean": row["utility_mean"],
                "variance": row["utility_variance"]
            }
        
        # Write the temporary results file
        temp_results_path = os.path.join(model_dir, "results_temp.json")
        with open(temp_results_path, 'w') as f:
            json.dump(temp_data, f)
        
        # Create a custom loader that works with our data structure
        def custom_loader(model_save_dir, category, measure):
            # Just return our preloaded dataframe
            return df, N_values_list
        
        # Monkey-patch the loader
        import create_exchange_rates_plots
        original_loader = create_exchange_rates_plots.load_thurstonian_results
        create_exchange_rates_plots.load_thurstonian_results = custom_loader
        
        try:
            # Generate plots
            figures_dict = plot_appendix_multi_model_average(
                results_dir=temp_dir,
                category='dummy',
                measure='dummy',
                canonical_X=canonical_X,
                plot_auxiliary_figures=True,
                model_include_list=[model],
                include_Xs=include_Xs,
                plot_scale=plot_scale,
                aggregator_plot_title=f"Exchange Rates for {prompt_config} ({model})",
                aggregator_plot_y_label="Exchange Rate",
                value_interpretation=value_interpretation
            )
        finally:
            # Restore original loader
            create_exchange_rates_plots.load_thurstonian_results = original_loader
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    if not figures_dict or 'aggregator_figure' not in figures_dict:
        raise RuntimeError("Failed to generate plots. Check that the data has multiple X values and N values.")
    
    # Save figures
    base_filename = f'{output_dir}/exchange_rates_{prompt_config}'
    figures_dict['aggregator_figure'].savefig(f'{base_filename}.pdf', bbox_inches='tight')
    figures_dict['aggregator_utility_style_bar_figure'].savefig(f'{base_filename}_utility_style_bar.pdf', bbox_inches='tight')
    figures_dict[f'{model}_aux_lnN_reg'].savefig(f'{base_filename}_regressions.pdf', bbox_inches='tight')
    
    print(f"Saved plots to {output_dir}")
    print(f"  - {base_filename}.pdf")
    print(f"  - {base_filename}_utility_style_bar.pdf")
    print(f"  - {base_filename}_regressions.pdf")
    
    return figures_dict


def main():
    parser = argparse.ArgumentParser(
        description='Generate exchange rates plots for stated preferences experiments'
    )
    parser.add_argument(
        '--results_dir',
        default='results_stated/',
        help='Base results directory (default: results_stated/)'
    )
    parser.add_argument(
        '--prompt_config',
        required=True,
        help='Prompt config name (e.g., gender_death_decision_original)'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o-mini',
        help='Model name (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--canonical_x',
        default=None,
        help='Reference X value for ratios (default: first X value)'
    )
    parser.add_argument(
        '--include_xs',
        nargs='+',
        default=None,
        help='List of X values to include (default: all)'
    )
    parser.add_argument(
        '--plot_scale',
        default='log',
        choices=['log', 'linear'],
        help='Plot scale (default: log)'
    )
    parser.add_argument(
        '--value_interpretation',
        default=None,
        choices=['positive', 'negative'],
        help='Value interpretation: positive (higher rates = more valued), negative (higher rates = less valued)'
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        help='Output directory for plots (default: same as results directory)'
    )
    
    args = parser.parse_args()
    
    create_plots_for_stated_preferences(
        results_dir=args.results_dir,
        prompt_config=args.prompt_config,
        model=args.model,
        canonical_X=args.canonical_x,
        include_Xs=args.include_xs,
        plot_scale=args.plot_scale,
        value_interpretation=args.value_interpretation,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

