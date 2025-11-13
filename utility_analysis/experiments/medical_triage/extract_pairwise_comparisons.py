#!/usr/bin/env python3

"""
Script to extract pairwise comparisons from result files to JSONL format.

Each line in the JSONL output contains:
- "A": dictionary for patient A (with only specified fields)
- "B": dictionary for patient B (with only specified fields)
- "decision": either "A" or "B"

The fields included are hardcoded at the top of the script and can be modified as needed.
"""

import json
import argparse
from typing import Dict, Any, List
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2


# Hardcoded list of fields to extract from each option
# Top-level fields to include directly
TOP_LEVEL_FIELDS = ['id', 'description']

# Fields to extract from patient_data
PATIENT_DATA_FIELDS = ['severity', 'group']

# Other nested fields to include
OTHER_FIELDS = ['factors']


def extract_option_fields(option: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only specified fields from an option dictionary.
    
    Args:
        option: Full option dictionary
        
    Returns:
        Dictionary with only the specified fields
    """
    extracted = {}
    
    # Extract top-level fields
    for field in TOP_LEVEL_FIELDS:
        if field in option:
            extracted[field] = option[field]
    
    # Extract patient_data fields
    if 'patient_data' in option:
        patient_data = {}
        for field in PATIENT_DATA_FIELDS:
            if field in option['patient_data']:
                patient_data[field] = option['patient_data'][field]
        if patient_data:
            extracted['patient_data'] = patient_data
    
    # Extract other fields
    for field in OTHER_FIELDS:
        if field in option:
            extracted[field] = option[field]
    
    return extracted


def extract_comparisons(results_path: str, output_path: str) -> None:
    """
    Extract all pairwise comparisons from a results file and save to JSONL.
    
    Args:
        results_path: Path to results JSON file
        output_path: Path to output JSONL file
    """
    # Load results
    print(f"Loading results from: {results_path}")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Get graph data which contains the edges (pairwise comparisons)
    graph_data = results.get('graph_data', {})
    edges = graph_data.get('edges', {})
    
    if not edges:
        print("Warning: No edges found in results file.")
        return
    
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
        
        # Extract only specified fields from options
        option_A_extracted = extract_option_fields(option_A)
        option_B_extracted = extract_option_fields(option_B)
        
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
    
    # Save to JSONL
    print(f"Saving to: {output_path}")
    with open(output_path, 'w') as f:
        for comparison in comparisons:
            f.write(json.dumps(comparison) + '\n')
    
    print(f"Successfully wrote {len(comparisons)} comparisons to {output_path}")
    
    # Print summary statistics
    decision_counts = {'A': 0, 'B': 0}
    for comp in comparisons:
        decision_counts[comp['decision']] += 1
    
    print("\nSummary:")
    print(f"  Total comparisons: {len(comparisons)}")
    print(f"  Decision A: {decision_counts['A']} ({100*decision_counts['A']/len(comparisons):.1f}%)")
    print(f"  Decision B: {decision_counts['B']} ({100*decision_counts['B']/len(comparisons):.1f}%)")


def analyze_decision_factors(jsonl_path: str) -> None:
    """
    Analyze which factors most affect the decision using logistic regression.
    
    Args:
        jsonl_path: Path to JSONL file with comparisons
    """
    print("\n" + "=" * 60)
    print("ANALYZING DECISION FACTORS")
    print("=" * 60)
    
    # Load JSONL file into a list
    comparisons = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            comparisons.append(json.loads(line))
    
    print(f"\nLoaded {len(comparisons)} comparisons from {jsonl_path}")
    
    # Convert to DataFrame
    rows = []
    for comp in comparisons:
        option_a = comp['A']
        option_b = comp['B']
        decision = comp['decision']
        
        # Extract patient data
        patient_a = option_a.get('patient_data', {})
        patient_b = option_b.get('patient_data', {})
        
        # Extract factors (e.g., Gender)
        factors_a = option_a.get('factors', {})
        factors_b = option_b.get('factors', {})
        
        row = {
            'choice': 1 if decision == 'A' else 0,  # 1 = chose A, 0 = chose B
            'severity_a': patient_a.get('severity'),
            'severity_b': patient_b.get('severity'),
            'group_a': patient_a.get('group'),
            'group_b': patient_b.get('group'),
        }
        
        # Add all factor values (dynamically detect factor types)
        for factor_name, factor_value in factors_a.items():
            row[f'{factor_name.lower()}_a'] = factor_value
        for factor_name, factor_value in factors_b.items():
            row[f'{factor_name.lower()}_b'] = factor_value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create difference variables
    # Severity difference (continuous)
    df['dsev'] = df['severity_a'] - df['severity_b']
    
    # Group difference - treat as categorical by creating dummy variables
    # We'll create indicators for whether A or B belongs to each group
    all_groups = sorted(df['group_a'].unique())
    reference_group = all_groups[0]  # Use first group as reference
    
    print(f"\nGroups found: {all_groups}")
    print(f"Using group {reference_group} as reference category")
    
    for group_num in all_groups:
        df[f'group_a_{group_num}'] = (df['group_a'] == group_num).astype(int)
        df[f'group_b_{group_num}'] = (df['group_b'] == group_num).astype(int)
    
    # For model, we'll use difference in group indicators
    # (having A in group X vs B in group X)
    # Skip the reference group to avoid multicollinearity
    group_diff_cols = []
    for group_num in all_groups[1:]:  # Skip reference group
        col_name = f'dgroup_{group_num}'
        df[col_name] = df[f'group_a_{group_num}'] - df[f'group_b_{group_num}']
        group_diff_cols.append(col_name)
    
    # Create factor difference variables (e.g., gender)
    factor_diff_cols = []
    
    # Detect which factors exist (e.g., 'gender')
    factor_columns = [col for col in df.columns if col.endswith('_a') and 
                     col not in ['severity_a', 'group_a'] and 
                     not col.startswith('group_a_')]
    
    for col_a in factor_columns:
        factor_name = col_a[:-2]  # Remove '_a' suffix
        col_b = f'{factor_name}_b'
        
        if col_b in df.columns:
            # Get unique values for this factor
            unique_values = set(df[col_a].unique()) | set(df[col_b].unique())
            unique_values.discard(None)
            unique_values = sorted(unique_values)
            
            print(f"\nFactor '{factor_name}' has values: {unique_values}")
            
            # Create difference variable for each value (except reference category)
            # Use first value alphabetically as reference
            for value in unique_values[1:]:  # Skip first as reference category
                diff_col = f'd{factor_name}_{value}'
                df[diff_col] = ((df[col_a] == value).astype(int) - 
                               (df[col_b] == value).astype(int))
                factor_diff_cols.append(diff_col)
    
    # Remove rows with missing values
    required_cols = ['choice', 'dsev'] + group_diff_cols + factor_diff_cols
    df_clean = df[required_cols].dropna()
    print(f"\nAfter removing NAs: {len(df_clean)} rows")
    
    # Fit full model (with all variables)
    X_full = sm.add_constant(df_clean[['dsev'] + group_diff_cols + factor_diff_cols])
    y = df_clean['choice']
    
    print("\n" + "=" * 60)
    print("FULL MODEL (all variables)")
    print("=" * 60)
    
    full_model = sm.Logit(y, X_full).fit(disp=0)
    print(full_model.summary())
    
    # Fit reduced model (without factor variables, only severity and group)
    X_reduced = sm.add_constant(df_clean[['dsev'] + group_diff_cols])
    
    print("\n" + "=" * 60)
    print("REDUCED MODEL (severity + group only)")
    print("=" * 60)
    
    reduced_model = sm.Logit(y, X_reduced).fit(disp=0)
    print(reduced_model.summary())
    
    # Likelihood ratio test for factor variables
    if factor_diff_cols:
        print("\n" + "=" * 60)
        print("LIKELIHOOD RATIO TEST FOR FACTOR VARIABLES")
        print("=" * 60)
        
        LR = 2 * (full_model.llf - reduced_model.llf)
        df_diff = full_model.df_model - reduced_model.df_model
        pval = 1 - chi2.cdf(LR, df_diff)
        
        print(f"\nTesting significance of: {', '.join(factor_diff_cols)}")
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
    print("\nPositive coefficients mean:")
    print("  - Higher value in A relative to B increases probability of choosing A")
    print("\nFor 'dsev' (severity difference):")
    print("  - Positive coef means higher severity in A increases chance of choosing A")
    print("  - Negative coef means higher severity in A decreases chance of choosing A")
    print("\nFor group differences:")
    print(f"  - Reference group: {reference_group}")
    print("  - Each dgroup_X coefficient shows effect of A being in group X (vs reference)")
    print("    relative to B being in group X (vs reference)")
    print("\nFor factor differences (e.g., gender):")
    print("  - Each coefficient shows effect relative to reference category")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract pairwise comparisons from results file to JSONL format and analyze decision factors."
    )
    parser.add_argument(
        "results_file",
        help="Path to results JSON file (e.g., results_utilities_*.json)"
    )
    parser.add_argument(
        "output_file",
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run statistical analysis on the extracted comparisons"
    )
    
    args = parser.parse_args()
    
    # Extract comparisons to JSONL
    extract_comparisons(args.results_file, args.output_file)
    
    # Optionally analyze the results
    if args.analyze:
        analyze_decision_factors(args.output_file)


if __name__ == "__main__":
    main()

