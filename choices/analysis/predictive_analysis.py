#!/usr/bin/env python3
"""
Script to extract pairwise comparisons from result files to JSONL format.
Works with both medical triage and stated preferences formats.

Each line in the JSONL output contains:
- "A": dictionary for option A (with only specified fields)
- "B": dictionary for option B (with only specified fields)
- "decision": either "A" or "B"

CONFIGURATION:
Field types for analysis are automatically read from results.graph.variables.
Each Variable.type determines how it's analyzed:
  - VariableType.NUMERICAL -> 'numerical' (simple difference)
  - VariableType.LOG_NUMERICAL -> 'log_numerical' (log difference for diminishing returns)
  - VariableType.CATEGORICAL -> 'categorical' (dummy variables)
  - VariableType.ORDINAL -> 'categorical' (treated as categorical)

To specify analysis type, use the appropriate Variable type when defining experiments:
  - numerical('severity', [1, 2, 3]) -> simple difference
  - log_numerical('N', [1, 2, 3, ...]) -> log difference
  - categorical('gender', ['male', 'female']) -> dummy variables
"""

import json
import argparse
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

from choices.results import ExperimentResults
from choices.variable import VariableType


# Field extraction is now automatic based on results.graph.variables
# No experiment-specific configuration needed here


def find_result_files(results_dir: str):
    """
    Find preference_graph and utility_model JSON files in the directory.
    
    Returns:
        (graph_file, model_file) tuple or (None, None) if not found
    """
    results_path = Path(results_dir)
    
    # Find preference_graph_*.json
    graph_files = list(results_path.glob("preference_graph_*.json"))
    if not graph_files:
        return None, None
    
    # Find utility_model_*.json
    model_files = list(results_path.glob("utility_model_*.json"))
    if not model_files:
        return None, None
    
    # Use the first file of each type (assuming only one)
    return str(graph_files[0]), str(model_files[0])


def extract_option_fields(option: Dict[str, Any], results: ExperimentResults) -> Dict[str, Any]:
    """
    Extract relevant fields from an option dictionary based on experiment variables.
    Completely generic - extracts all fields that correspond to variables.
    
    Args:
        option: Full option dictionary (can be ExperimentOption.to_dict() or dict)
        results: ExperimentResults object to determine which fields to extract
        
    Returns:
        Dictionary with extracted fields
    """
    extracted = {}
    
    # Always extract id and description
    if 'id' in option:
        extracted['id'] = option['id']
    if 'description' in option:
        extracted['description'] = option['description']
    
    # Extract all fields that correspond to variables (variables are at top level)
    for var_name in results.graph.variables.keys():
        if var_name in option:
            extracted[var_name] = option[var_name]
    
    return extracted


def extract_comparisons_from_graph(results: ExperimentResults) -> List[Dict[str, Any]]:
    """
    Extract all pairwise comparisons directly from the preference graph.
    
    Args:
        results: ExperimentResults object
    
    Returns:
        List of comparison dictionaries with 'A', 'B', and 'decision' keys
    """
    # Get edges from graph
    edges = results.graph.edges
    
    if not edges:
        print("Warning: No edges found in graph.")
        return []
    
    print(f"Found {len(edges)} edges in graph data")
    
    # Extract all individual comparisons
    comparisons = []
    total_responses = 0
    
    for edge_key, edge_data in edges.items():
        option_A = edge_data['option_A']
        option_B = edge_data['option_B']
        aux_data = edge_data.get('aux_data', {})
        
        # Get parsed responses from both original and flipped prompts
        original_parsed = aux_data.get('original_parsed', [])
        flipped_parsed = aux_data.get('flipped_parsed', [])
        
        # If not in aux_data, try alternative locations
        if not original_parsed and not flipped_parsed:
            # Try to get from original_responses and flipped_responses
            original_responses = aux_data.get('original_responses', [])
            flipped_responses = aux_data.get('flipped_responses', [])
            original_parsed = original_responses
            flipped_parsed = flipped_responses
        
        # Extract relevant fields from options
        # Convert ExperimentOption to dict if needed
        if hasattr(option_A, 'to_dict'):
            option_A = option_A.to_dict()
        if hasattr(option_B, 'to_dict'):
            option_B = option_B.to_dict()
        
        option_A_extracted = extract_option_fields(option_A, results)
        option_B_extracted = extract_option_fields(option_B, results)
        
        # Process original direction responses (A shown first, B shown second)
        # 'A' means A was chosen, 'B' means B was chosen
        for response in original_parsed:
            if response in ['A', 'B']:
                comparisons.append({
                    'A': option_A_extracted,
                    'B': option_B_extracted,
                    'decision': response
                })
                total_responses += 1
        
        # Process flipped direction responses (B shown first as "A", A shown second as "B")
        # To balance the dataset, we swap A and B for flipped responses
        # This way the response labels 'A' and 'B' correctly refer to the swapped positions
        for response in flipped_parsed:
            if response in ['A', 'B']:
                comparisons.append({
                    'A': option_B_extracted,  # Swap: B becomes A
                    'B': option_A_extracted,  # Swap: A becomes B
                    'decision': response       # Keep decision as-is (now refers to swapped positions)
                })
                total_responses += 1
    
    print(f"Extracted {len(comparisons)} individual comparisons from {total_responses} responses")
    
    # Print summary statistics
    decision_counts = {'A': 0, 'B': 0}
    for comp in comparisons:
        decision_counts[comp['decision']] += 1
    
    print("\nSummary:")
    print(f"  Total comparisons: {len(comparisons)}")
    if len(comparisons) > 0:
        print(f"  Decision A: {decision_counts['A']} ({100*decision_counts['A']/len(comparisons):.1f}%)")
        print(f"  Decision B: {decision_counts['B']} ({100*decision_counts['B']/len(comparisons):.1f}%)")
    
    return comparisons


def get_field_types_from_results(results: ExperimentResults) -> Dict[str, str]:
    """
    Extract field types from experiment results.
    
    Maps VariableType to analysis type string:
    - CATEGORICAL -> 'categorical'
    - NUMERICAL -> 'numerical'
    - LOG_NUMERICAL -> 'log_numerical'
    - ORDINAL -> 'categorical'
    
    Returns:
        Dict mapping field name to analysis type
    """
    field_types = {}
    
    for var_name, variable in results.graph.variables.items():
        if variable.type == VariableType.CATEGORICAL:
            field_types[var_name] = 'categorical'
        elif variable.type == VariableType.NUMERICAL:
            field_types[var_name] = 'numerical'
        elif variable.type == VariableType.LOG_NUMERICAL:
            field_types[var_name] = 'log_numerical'
        elif variable.type == VariableType.ORDINAL:
            field_types[var_name] = 'categorical'  # Treat ordinal as categorical
        else:
            # Default fallback
            field_types[var_name] = 'categorical'
    
    return field_types


def analyze_decision_factors(comparisons: List[Dict[str, Any]], results: ExperimentResults) -> None:
    """
    Analyze which factors most affect the decision using logistic regression.
    Completely generic - works with any experiment format based on variables defined in results.
    
    Args:
        comparisons: List of comparison dictionaries with 'A', 'B', and 'decision' keys
        results: ExperimentResults object containing variable definitions
    """
    print("\n" + "=" * 60)
    print("ANALYZING DECISION FACTORS")
    print("=" * 60)
    
    print(f"\nAnalyzing {len(comparisons)} comparisons")
    
    if not results.graph.variables:
        print("Error: No variables found in results. Cannot analyze.")
        return
    
    print(f"\nVariables defined in experiment:")
    for var_name, variable in results.graph.variables.items():
        print(f"  {var_name}: {variable.type.value} ({len(variable)} values)")
    
    # Convert to DataFrame - extract all fields from variables
    rows = []
    for comp in comparisons:
        option_a = comp['A']
        option_b = comp['B']
        decision = comp['decision']
        
        row = {'choice': 1 if decision == 'A' else 0}
        
        # Extract all fields that correspond to variables
        for var_name in results.graph.variables.keys():
            # Variables are stored at top level in options
            if var_name in option_a:
                row[f'{var_name}_a'] = option_a.get(var_name)
                row[f'{var_name}_b'] = option_b.get(var_name)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Get field types from results
    field_types = get_field_types_from_results(results)
    
    # Detect factor field - use categorical variables as factors
    # Prefer factor_value if present, otherwise first categorical variable
    factor_field_name = None
    if 'factor_value' in results.graph.variables:
        if results.graph.variables['factor_value'].type in [VariableType.CATEGORICAL, VariableType.ORDINAL]:
            factor_field_name = 'factor_value'
    
    if factor_field_name is None:
        # Check other categorical variables
        for var_name, variable in results.graph.variables.items():
            if variable.type in [VariableType.CATEGORICAL, VariableType.ORDINAL]:
                col_a = f'{var_name}_a'
                if col_a in df.columns and df[col_a].notna().any():
                    factor_field_name = var_name
                    break
    
    if not field_types:
        print("  Error: No variables found in results. Cannot determine field types.")
        return
    
    print(f"\nField types: {field_types}")
    if factor_field_name:
        print(f"Factor field detected: {factor_field_name}")
    
    # Create difference variables (same logic for both formats)
    numerical_vars = []
    categorical_var_cols = {}  # field_name -> list of diff columns
    factor_var_cols = []  # Factor variables (to be tested)
    
    for field_name, field_type in field_types.items():
        col_a = f'{field_name}_a'
        col_b = f'{field_name}_b'
        
        if col_a not in df.columns or col_b not in df.columns:
            continue  # Field not present in data
        
        # Check if this field is the factor being tested
        is_factor = (field_name == factor_field_name)
        
        if field_type == 'numerical':
            # Simple difference
            diff_col = f'd{field_name}'
            df[diff_col] = df[col_a] - df[col_b]
            numerical_vars.append(diff_col)
            print(f"\nField '{field_name}': numerical (using difference)")
        
        elif field_type == 'log_numerical':
            # Log difference
            diff_col = f'dlog{field_name}'
            df[diff_col] = pd.Series([
                0 if na == nb else (np.log(na) - np.log(nb) if na > 0 and nb > 0 else 0)
                for na, nb in zip(df[col_a], df[col_b])
            ])
            numerical_vars.append(diff_col)
            print(f"\nField '{field_name}': log_numerical (using log difference, for diminishing returns)")
        
        elif field_type == 'categorical':
            # Create dummy variables
            unique_values = sorted(set(df[col_a].dropna().unique()) | set(df[col_b].dropna().unique()))
            if not unique_values:
                continue
            
            # Skip if only one unique value (no variation)
            if len(unique_values) == 1:
                print(f"\nField '{field_name}': categorical - SKIPPED (only one value: {unique_values[0]})")
                continue
            
            reference_value = unique_values[0]
            
            print(f"\nField '{field_name}': categorical" + (" (factor being tested)" if is_factor else ""))
            print(f"  Values: {unique_values}")
            print(f"  Reference: {reference_value}")
            
            # Create indicators for each value
            for val in unique_values:
                df[f'{field_name}_a_{val}'] = (df[col_a] == val).astype(int)
                df[f'{field_name}_b_{val}'] = (df[col_b] == val).astype(int)
            
            # Create difference variables (skip reference)
            diff_cols = []
            for val in unique_values[1:]:
                diff_col = f'd{field_name}_{val}'
                df[diff_col] = df[f'{field_name}_a_{val}'] - df[f'{field_name}_b_{val}']
                # Check if this difference variable is constant (all zeros or all same value)
                if df[diff_col].nunique() <= 1:
                    print(f"    Skipping {diff_col} (constant)")
                    continue
                diff_cols.append(diff_col)
            
            # Only add if we have valid difference columns
            if diff_cols:
                # Separate factors from control variables
                if is_factor:
                    factor_var_cols.extend(diff_cols)
                else:
                    categorical_var_cols[field_name] = diff_cols
    
    
    # Build list of all predictors (numerical + categorical controls)
    all_numerical = numerical_vars
    all_categorical = [col for cols in categorical_var_cols.values() for col in cols]
    control_vars = all_numerical + all_categorical
    
    # Remove rows with missing values
    required_cols = ['choice']
    required_cols.extend(control_vars)
    required_cols.extend(factor_var_cols)
    
    df_clean = df[required_cols].dropna()
    print(f"\nAfter removing NAs: {len(df_clean)} rows")
    
    # Fit full model (all variables)
    all_predictors = control_vars + factor_var_cols
    
    if not all_predictors:
        print("\nWarning: No predictors available. Cannot fit model.")
        return
    
    # Remove constant columns (perfect multicollinearity)
    constant_cols = []
    for col in all_predictors:
        if col in df_clean.columns and df_clean[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"\nRemoving constant columns: {constant_cols}")
        all_predictors = [p for p in all_predictors if p not in constant_cols]
        control_vars = [p for p in control_vars if p not in constant_cols]
        factor_var_cols = [p for p in factor_var_cols if p not in constant_cols]
    
    if not all_predictors:
        print("\nWarning: No predictors available after removing constants. Cannot fit model.")
        return
    
    # Check for perfect multicollinearity by computing rank
    X_full = sm.add_constant(df_clean[all_predictors])
    y = df_clean['choice']
    
    # Check matrix rank to detect perfect multicollinearity
    try:
        rank = np.linalg.matrix_rank(X_full.values)
        if rank < X_full.shape[1]:
            print(f"\nWarning: Design matrix rank ({rank}) < number of columns ({X_full.shape[1]})")
            print("This indicates perfect multicollinearity. Attempting to remove redundant columns...")
            
            # Use QR decomposition to identify redundant columns
            Q, R = np.linalg.qr(X_full.values)
            # Find columns that are linearly dependent (near-zero diagonal in R)
            tol = 1e-10
            redundant = []
            for i in range(R.shape[1]):
                if abs(R[i, i]) < tol:
                    redundant.append(X_full.columns[i])
            
            if redundant:
                print(f"Removing redundant columns: {redundant}")
                all_predictors = [p for p in all_predictors if p not in redundant]
                control_vars = [p for p in control_vars if p not in redundant]
                factor_var_cols = [p for p in factor_var_cols if p not in redundant]
                X_full = sm.add_constant(df_clean[all_predictors])
    except Exception as e:
        print(f"\nWarning: Could not check matrix rank: {e}")
        print("Proceeding with model fit...")
    
    print("\n" + "=" * 60)
    print("FULL MODEL (all variables)")
    print("=" * 60)
    
    full_model = sm.Logit(y, X_full).fit(disp=0)
    print(full_model.summary())
    
    # Fit reduced model (without factor variables)
    reduced_predictors = control_vars  # All configured fields, no factors
    
    if not reduced_predictors:
        print("\nWarning: No control variables available. Skipping reduced model.")
        reduced_model = None
    else:
        X_reduced = sm.add_constant(df_clean[reduced_predictors])
        
        print("\n" + "=" * 60)
        print("REDUCED MODEL (without factors)")
        print("=" * 60)
        
        reduced_model = sm.Logit(y, X_reduced).fit(disp=0)
        print(reduced_model.summary())
    
    # Likelihood ratio test for factor variables
    if factor_var_cols and reduced_model is not None:
        print("\n" + "=" * 60)
        print("LIKELIHOOD RATIO TEST FOR FACTOR VARIABLES")
        print("=" * 60)
        
        LR = 2 * (full_model.llf - reduced_model.llf)
        df_diff = full_model.df_model - reduced_model.df_model
        pval = 1 - chi2.cdf(LR, df_diff)
        
        print(f"\nTesting significance of: {', '.join(factor_var_cols)}")
        print(f"LR statistic: {LR:.3f}")
        print(f"Degrees of freedom: {df_diff}")
        print(f"P-value: {pval:.3g}")
        
        if pval < 0.001:
            print("*** Highly significant (p < 0.001)")
        elif pval < 0.01:
            print("** Very significant (p < 0.01)")
        elif pval < 0.05:
            print("* Significant (p < 0.05)")
        else:
            print("Not significant (p >= 0.05)")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    # Extract key coefficients for interpretation
    coef_names = full_model.params.index.tolist()
    coef_values = full_model.params.values
    
    print("\n1. DIRECTION (sign of coefficient):")
    print("   Positive coef → higher value in A (vs B) increases probability of choosing A")
    print("   Negative coef → higher value in A (vs B) decreases probability of choosing A")
    
    print("\n2. MAGNITUDE (as odds ratios):")
    print("   exp(coefficient) = odds ratio")
    
    # Show numerical variables
    numerical_coefs = [(name, full_model.params[name]) for name in coef_names 
                       if name in all_numerical and name != 'const']
    if numerical_coefs:
        print("\n   Numerical variables:")
        for var_name, coef in numerical_coefs:
            or_val = np.exp(coef)
            
            # Check if it's a log variable
            if var_name.startswith('dlog'):
                field_name = var_name[4:]  # Remove 'dlog' prefix
                double_effect = np.exp(coef * np.log(2))
                tenfold_effect = np.exp(coef * np.log(10))
                print(f"   {var_name} = {coef:.3f}")
                print(f"     → Doubling {field_name} multiplies odds by {double_effect:.2f}x")
                print(f"     → 10x increase in {field_name} multiplies odds by {tenfold_effect:.2f}x")
            else:
                field_name = var_name[1:] if var_name.startswith('d') else var_name
                print(f"   {var_name} = {coef:.3f} → OR = {or_val:.2f}")
                print(f"     → Each 1-unit increase in {field_name} multiplies odds by {or_val:.2f}x")
                if abs(coef) > 0.1:
                    print(f"     → +2 units in {field_name} has {or_val**2:.2f}x odds")
    
    # Show categorical control variables
    categorical_coefs = [(name, full_model.params[name]) for name in coef_names 
                        if name in all_categorical and name != 'const']
    if categorical_coefs:
        # Group by base field name
        field_groups = {}
        for name, coef in categorical_coefs:
            # Extract field name (e.g., 'group' from 'dgroup_2')
            if '_' in name:
                base_name = name.split('_')[0][1:]  # Remove 'd' prefix
                if base_name not in field_groups:
                    field_groups[base_name] = []
                field_groups[base_name].append((name, coef))
        
        for field_name, coefs in field_groups.items():
            print(f"\n   {field_name.capitalize()} effects:")
            for name, coef in coefs:
                value = '_'.join(name.split('_')[1:])  # Get the value (e.g., '2' from 'dgroup_2')
                or_val = np.exp(coef)
                if coef > 0:
                    print(f"   {name} = {coef:.3f} → OR = {or_val:.2f} ({field_name} {value} favored by {or_val:.2f}x)")
                else:
                    print(f"   {name} = {coef:.3f} → OR = {or_val:.2f} ({field_name} {value} penalized, {1/or_val:.2f}x less likely)")
    
    # Factor effects (categorical variables from factors)
    factor_coefs = [(name, full_model.params[name], np.exp(full_model.params[name])) 
                    for name in coef_names if name in factor_var_cols]
    if factor_coefs:
        print(f"\n   Factor effects:")
        # Group by factor name
        factor_groups = {}
        for name, coef, or_val in factor_coefs:
            # Extract factor name (everything before last '_')
            parts = name.split('_')
            if len(parts) >= 2:
                factor_name = '_'.join(parts[:-1])[1:]  # Remove 'd' prefix
                if factor_name not in factor_groups:
                    factor_groups[factor_name] = []
                factor_groups[factor_name].append((name, coef, or_val))
        
        for factor_name, coefs in factor_groups.items():
            print(f"   {factor_name.capitalize()}:")
            for name, coef, or_val in coefs:
                value = name.split('_')[-1]  # Get the value (last part)
                if abs(coef) < 0.001:
                    print(f"     {name} = {coef:.3f} → OR = {or_val:.3f} (essentially no effect)")
                elif coef > 0:
                    print(f"     {name} = {coef:.3f} → OR = {or_val:.2f} ({value} favored by {or_val:.2f}x)")
                else:
                    print(f"     {name} = {coef:.3f} → OR = {or_val:.2f} ({value} penalized, {1/or_val:.2f}x less likely)")
    
    print("\n3. PRACTICAL SIGNIFICANCE:")
    print("   Compare coefficient magnitudes to see what matters most")
    print("   Small coefficients (<0.1) = weak effect")
    print("   Medium coefficients (0.1-0.5) = moderate effect")
    print("   Large coefficients (>0.5) = strong effect")
    
    print("\n4. BASELINE (when all differences = 0):")
    const_coef = full_model.params['const']
    baseline_prob = 1 / (1 + np.exp(-const_coef))
    print(f"   Constant = {const_coef:.3f}")
    print(f"   → When A and B are identical, P(choose A) = {baseline_prob:.1%}")
    if abs(baseline_prob - 0.5) > 0.05:
        if baseline_prob > 0.5:
            print(f"   → Bias toward choosing first option (A)")
        else:
            print(f"   → Bias toward choosing second option (B)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze decision factors from preference graph using logistic regression."
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing preference_graph_*.json and utility_model_*.json files"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional: Path to save extracted comparisons as JSONL file"
    )
    
    args = parser.parse_args()
    
    # Find and load result files
    graph_file, model_file = find_result_files(args.results_dir)
    if graph_file is None or model_file is None:
        raise FileNotFoundError(
            f"Could not find preference_graph_*.json and utility_model_*.json files in {args.results_dir}"
        )
    
    # Extract suffix from filenames
    graph_path = Path(graph_file)
    model_path = Path(model_file)
    
    graph_suffix = graph_path.stem.replace("preference_graph_", "")
    model_suffix = model_path.stem.replace("utility_model_", "")
    
    if graph_suffix != model_suffix:
        raise ValueError(
            f"Suffix mismatch: graph has '{graph_suffix}', model has '{model_suffix}'"
        )
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = ExperimentResults.load(args.results_dir, graph_suffix)
    
    # Extract comparisons directly from graph
    comparisons = extract_comparisons_from_graph(results)
    
    if not comparisons:
        print("No comparisons found. Cannot analyze.")
        return
    
    # Optionally save to JSONL
    if args.output:
        print(f"\nSaving comparisons to: {args.output}")
        with open(args.output, 'w') as f:
            for comparison in comparisons:
                f.write(json.dumps(comparison) + '\n')
        print(f"Successfully wrote {len(comparisons)} comparisons to {args.output}")
    
    # Analyze the results
    analyze_decision_factors(comparisons, results=results)


if __name__ == "__main__":
    main()

