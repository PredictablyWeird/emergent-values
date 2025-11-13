#!/usr/bin/env python3

"""
Script to extract pairwise comparisons from result files to JSONL format.
Works with both medical triage and stated preferences formats.

Each line in the JSONL output contains:
- "A": dictionary for option A (with only specified fields)
- "B": dictionary for option B (with only specified fields)
- "decision": either "A" or "B"

Field selection is configured via TRIAGE_FIELDS and STATED_PREF_FIELDS:
- TRIAGE_FIELDS: Configure medical triage fields (severity, group, factors, etc.)
- STATED_PREF_FIELDS: Configure stated preferences fields (N, X, factor)

Modify these config dictionaries at the top of the script to customize field extraction.
"""

import json
import argparse
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2


# ========== FIELD SELECTION CONFIGURATION ==========
# Modify these dictionaries to customize which fields are extracted

# Medical Triage Format: Fields to extract
TRIAGE_FIELDS = {
    'top_level': ['id', 'description'],
    'patient_data': ['severity', 'group', 'primary_condition'],
    'other': ['factors']  # Nested fields like factors
}
# Example: To include more patient_data fields, add them to the list:
# 'patient_data': ['severity', 'group', 'primary_condition', 'patient_id', 'sofa']

# Stated Preferences Format: Fields to extract
STATED_PREF_FIELDS = {
    'top_level': ['id', 'description'],
    'direct': ['N', 'X', 'factor']  # Direct top-level fields
}
# Example: All fields at the option level go in 'direct' or 'top_level'


def extract_option_fields(option: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only specified fields from an option dictionary.
    Works with both medical triage and stated preferences formats.
    Uses TRIAGE_FIELDS and STATED_PREF_FIELDS configurations.
    
    Args:
        option: Full option dictionary
        
    Returns:
        Dictionary with only the specified fields
    """
    extracted = {}
    
    # Detect format
    is_medical_triage = 'patient_data' in option
    is_stated_preferences = 'N' in option and 'X' in option
    
    if is_medical_triage:
        # Extract top-level fields
        for field in TRIAGE_FIELDS['top_level']:
            if field in option:
                extracted[field] = option[field]
        
        # Extract patient_data fields
        if 'patient_data' in option:
            patient_data = {}
            for field in TRIAGE_FIELDS['patient_data']:
                if field in option['patient_data']:
                    patient_data[field] = option['patient_data'][field]
            if patient_data:
                extracted['patient_data'] = patient_data
        
        # Extract other nested fields (like factors)
        for field in TRIAGE_FIELDS['other']:
            if field in option:
                extracted[field] = option[field]
    
    elif is_stated_preferences:
        # Extract top-level fields
        for field in STATED_PREF_FIELDS['top_level']:
            if field in option:
                extracted[field] = option[field]
        
        # Extract direct fields
        for field in STATED_PREF_FIELDS['direct']:
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
    Works with both medical triage format and stated preferences format.
    
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
    
    # Detect format (medical triage vs stated preferences)
    first_comp = comparisons[0]
    is_medical_triage = 'patient_data' in first_comp['A']
    is_stated_preferences = 'N' in first_comp['A']
    
    if is_medical_triage:
        print("Format detected: Medical Triage (severity + group)")
    elif is_stated_preferences:
        print("Format detected: Stated Preferences (N + X)")
    else:
        print("Warning: Unknown format")
    
    # Convert to DataFrame
    rows = []
    for comp in comparisons:
        option_a = comp['A']
        option_b = comp['B']
        decision = comp['decision']
        
        row = {'choice': 1 if decision == 'A' else 0}
        
        if is_medical_triage:
            # Extract patient data
            patient_a = option_a.get('patient_data', {})
            patient_b = option_b.get('patient_data', {})
            
            row['severity_a'] = patient_a.get('severity')
            row['severity_b'] = patient_b.get('severity')
            row['group_a'] = patient_a.get('group')
            row['group_b'] = patient_b.get('group')
            
            # Extract factors (e.g., Gender)
            factors_a = option_a.get('factors', {})
            factors_b = option_b.get('factors', {})
            
            for factor_name, factor_value in factors_a.items():
                row[f'{factor_name.lower()}_a'] = factor_value
            for factor_name, factor_value in factors_b.items():
                row[f'{factor_name.lower()}_b'] = factor_value
        
        elif is_stated_preferences:
            # Extract N and X (factor value)
            row['N_a'] = option_a.get('N')
            row['N_b'] = option_b.get('N')
            row['X_a'] = option_a.get('X')
            row['X_b'] = option_b.get('X')
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create difference variables based on format
    if is_medical_triage:
        # Severity difference (continuous)
        df['dsev'] = df['severity_a'] - df['severity_b']
        
        # Group difference - treat as categorical by creating dummy variables
        all_groups = sorted(df['group_a'].unique())
        reference_group = all_groups[0]  # Use first group as reference
        
        print(f"\nGroups found: {all_groups}")
        print(f"Using group {reference_group} as reference category")
        
        for group_num in all_groups:
            df[f'group_a_{group_num}'] = (df['group_a'] == group_num).astype(int)
            df[f'group_b_{group_num}'] = (df['group_b'] == group_num).astype(int)
        
        # For model, we'll use difference in group indicators
        # Skip the reference group to avoid multicollinearity
        group_diff_cols = []
        for group_num in all_groups[1:]:  # Skip reference group
            col_name = f'dgroup_{group_num}'
            df[col_name] = df[f'group_a_{group_num}'] - df[f'group_b_{group_num}']
            group_diff_cols.append(col_name)
        
        # Detect factor columns (gender, etc.)
        factor_columns = [col for col in df.columns if col.endswith('_a') and 
                         col not in ['severity_a', 'group_a'] and 
                         not col.startswith('group_a_')]
        
        main_var = 'dsev'
        control_vars = group_diff_cols
        
    elif is_stated_preferences:
        # N difference - use both linear and log for diminishing returns
        df['dN'] = df['N_a'] - df['N_b']
        df['dlogN'] = pd.Series([
            0 if na == nb else (np.log(na) - np.log(nb) if na > 0 and nb > 0 else 0)
            for na, nb in zip(df['N_a'], df['N_b'])
        ])
        
        print("\nNumber of patients (N) variable created")
        print(f"  Using both dN (linear) and dlogN (logarithmic) for diminishing returns")
        
        # Detect factor columns (X values)
        factor_columns = ['X']  # In stated preferences, X is the factor
        
        main_var = 'dlogN'  # Use log for main analysis (diminishing returns)
        control_vars = []  # No group controls in stated preferences
        group_diff_cols = []
    
    # Create factor difference variables
    factor_diff_cols = []
    
    if is_medical_triage:
        # Medical triage: factors are in separate columns with names
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
    
    elif is_stated_preferences:
        # Stated preferences: X is the factor value
        if 'X_a' in df.columns and 'X_b' in df.columns:
            unique_values = set(df['X_a'].unique()) | set(df['X_b'].unique())
            unique_values.discard(None)
            unique_values = sorted(unique_values)
            
            print(f"\nFactor (X) has values: {unique_values}")
            
            # Create difference variable for each value (except reference category)
            for value in unique_values[1:]:  # Skip first as reference category
                diff_col = f'dX_{value}'
                df[diff_col] = ((df['X_a'] == value).astype(int) - 
                               (df['X_b'] == value).astype(int))
                factor_diff_cols.append(diff_col)
    
    # Remove rows with missing values
    required_cols = ['choice', main_var] + control_vars + factor_diff_cols
    df_clean = df[required_cols].dropna()
    print(f"\nAfter removing NAs: {len(df_clean)} rows")
    
    # Fit full model (with all variables)
    X_full = sm.add_constant(df_clean[[main_var] + control_vars + factor_diff_cols])
    y = df_clean['choice']
    
    print("\n" + "=" * 60)
    print("FULL MODEL (all variables)")
    print("=" * 60)
    
    full_model = sm.Logit(y, X_full).fit(disp=0)
    print(full_model.summary())
    
    # Fit reduced model (without factor variables)
    X_reduced = sm.add_constant(df_clean[[main_var] + control_vars])
    
    print("\n" + "=" * 60)
    if is_medical_triage:
        print("REDUCED MODEL (severity + group only)")
    elif is_stated_preferences:
        print("REDUCED MODEL (N only, no factor)")
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
    
    if is_medical_triage:
        print("\nFor 'dsev' (severity difference):")
        print("  - Positive coef means higher severity in A increases chance of choosing A")
        print("  - Negative coef means higher severity in A decreases chance of choosing A")
        print("\nFor group differences:")
        print(f"  - Reference group: {reference_group}")
        print("  - Each dgroup_X coefficient shows effect of A being in group X (vs reference)")
        print("    relative to B being in group X (vs reference)")
        print("\nFor factor differences (e.g., gender):")
        print("  - Each coefficient shows effect relative to reference category")
    
    elif is_stated_preferences:
        print("\nFor 'dlogN' (log number of patients difference):")
        print("  - Positive coef means more patients in A increases chance of choosing A")
        print("  - Uses logarithm to capture diminishing returns (saving 10 vs 9 people")
        print("    matters more than saving 100 vs 99)")
        print("  - Coefficient represents effect of doubling the number of patients")
        print("\nFor factor differences (dX_...):")
        # Get unique X values from the original dataframe
        if 'X_a' in df.columns:
            unique_x_values = sorted(set(df['X_a'].unique()) | set(df['X_b'].unique()))
            reference_x = unique_x_values[0]
            print(f"  - Reference category: {reference_x}")
            print(f"  - Each dX_Y coefficient shows effect of Y vs {reference_x}")
    
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

