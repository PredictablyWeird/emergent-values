#!/usr/bin/env python3

"""
Script to extract pairwise comparisons from result files to JSONL format.
Works with both medical triage and stated preferences formats.

Each line in the JSONL output contains:
- "A": dictionary for option A (with only specified fields)
- "B": dictionary for option B (with only specified fields)
- "decision": either "A" or "B"

CONFIGURATION (at top of script):
1. Field Selection:
   - TRIAGE_FIELDS / STATED_PREF_FIELDS: Which fields to extract
2. Field Types (for analysis):
   - TRIAGE_FIELD_TYPES / STATED_PREF_FIELD_TYPES: How to use each field
   - Options: 'numerical', 'categorical', 'log_numerical'

Examples:
  'severity': 'numerical'       → continuous difference (higher severity = more priority)
  'group': 'categorical'         → dummy variables (no inherent ordering)
  'N': 'log_numerical'          → log difference (captures diminishing returns)
"""

import json
import argparse
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2


# ========== FIELD SELECTION CONFIGURATION ==========
# Modify these dictionaries to customize which fields are extracted and how they're used

# STEP 1: Specify which fields to EXTRACT to JSONL file
# Medical Triage Format: Fields to extract
TRIAGE_FIELDS = {
    'top_level': ['id', 'description'],
    'patient_data': ['severity', 'group', 'sofa'],  # Fields from patient_data object
    'other': ['factors']  # Nested fields like factors (always categorical)
}

# STEP 2: Specify how to USE each field in ANALYSIS
# Medical Triage: How to use each field for prediction
# Options: 'categorical', 'numerical', 'log_numerical'
TRIAGE_FIELD_TYPES = {
    'severity': 'numerical',      # Continuous variable
    #'group': 'categorical',        # Dummy variables (no ordering)
    'sofa': 'numerical',          # Continuous variable
    # 'factors' (like gender) are always categorical
}
# IMPORTANT: Fields in TRIAGE_FIELD_TYPES must also be in TRIAGE_FIELDS['patient_data']
# Example: To add 'age':
#   1. Add 'age' to TRIAGE_FIELDS['patient_data']
#   2. Add 'age': 'numerical' to TRIAGE_FIELD_TYPES

# Stated Preferences Format: Fields to extract
STATED_PREF_FIELDS = {
    'top_level': ['id', 'description'],
    'direct': ['N', 'X', 'factor']  # Direct top-level fields
}

# How to use each field for prediction (stated preferences)
STATED_PREF_FIELD_TYPES = {
    'N': 'log_numerical',    # Log transform for diminishing returns
    'X': 'categorical',      # Factor values (no ordering)
}

# Validation: Ensure field types are defined for fields that will be extracted
def validate_config():
    """Validate that FIELD_TYPES and FIELDS configurations are consistent."""
    # Check triage fields
    patient_data_fields = TRIAGE_FIELDS.get('patient_data', [])
    for field in patient_data_fields:
        if field not in TRIAGE_FIELD_TYPES and field not in ['patient_id', 'primary_condition']:
            print(f"WARNING: Field '{field}' in TRIAGE_FIELDS['patient_data'] but not in TRIAGE_FIELD_TYPES")
            print(f"         This field will be extracted but not used in analysis.")
    
    for field in TRIAGE_FIELD_TYPES.keys():
        if field not in patient_data_fields:
            print(f"WARNING: Field '{field}' in TRIAGE_FIELD_TYPES but not in TRIAGE_FIELDS['patient_data']")
            print(f"         This field will not be available for analysis.")
    
    # Check stated pref fields
    direct_fields = STATED_PREF_FIELDS.get('direct', [])
    for field in STATED_PREF_FIELD_TYPES.keys():
        if field not in direct_fields:
            print(f"WARNING: Field '{field}' in STATED_PREF_FIELD_TYPES but not in STATED_PREF_FIELDS['direct']")
            print(f"         This field will not be available for analysis.")

validate_config()


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
            # Extract patient data - dynamically extract all fields from TRIAGE_FIELD_TYPES
            patient_a = option_a.get('patient_data', {})
            patient_b = option_b.get('patient_data', {})
            
            # Extract all fields specified in TRIAGE_FIELD_TYPES
            for field_name in TRIAGE_FIELD_TYPES.keys():
                row[f'{field_name}_a'] = patient_a.get(field_name)
                row[f'{field_name}_b'] = patient_b.get(field_name)
            
            # Extract factors (e.g., Gender)
            factors_a = option_a.get('factors', {})
            factors_b = option_b.get('factors', {})
            
            for factor_name, factor_value in factors_a.items():
                row[f'{factor_name.lower()}_a'] = factor_value
            for factor_name, factor_value in factors_b.items():
                row[f'{factor_name.lower()}_b'] = factor_value
        
        elif is_stated_preferences:
            # Extract all fields specified in STATED_PREF_FIELD_TYPES
            for field_name in STATED_PREF_FIELD_TYPES.keys():
                row[f'{field_name}_a'] = option_a.get(field_name)
                row[f'{field_name}_b'] = option_b.get(field_name)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Get field type configuration based on format
    if is_medical_triage:
        field_types = TRIAGE_FIELD_TYPES
        factor_field_name = None  # Factors come from nested 'factors' field, not configured
    elif is_stated_preferences:
        field_types = STATED_PREF_FIELD_TYPES
        factor_field_name = 'X'  # X is the factor being tested
    
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
            unique_values = sorted(set(df[col_a].unique()) | set(df[col_b].unique()))
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
                diff_cols.append(diff_col)
            
            # Separate factors from control variables
            if is_factor:
                factor_var_cols.extend(diff_cols)
            else:
                categorical_var_cols[field_name] = diff_cols
    
    # Detect additional factor columns (medical triage: factors from nested 'factors' field)
    if is_medical_triage:
        factor_columns = [col for col in df.columns if col.endswith('_a') and 
                         col not in [f'{f}_a' for f in field_types.keys()] and
                         not any(col.startswith(f'{f}_a_') for f in field_types.keys())]
    else:
        factor_columns = []
    
    # Build list of all predictors (numerical + categorical controls)
    all_numerical = numerical_vars
    all_categorical = [col for cols in categorical_var_cols.values() for col in cols]
    control_vars = all_numerical + all_categorical
    
    # Process additional factor columns (medical triage: nested factors like Gender)
    # For stated preferences, factor columns were already created in main loop
    factor_diff_cols = list(factor_var_cols)  # Start with factors from configured fields
    
    # Medical triage: add factors from nested 'factors' field
    for col_a in factor_columns:
        factor_name = col_a[:-2]  # Remove '_a' suffix
        col_b = f'{factor_name}_b'
        
        if col_b in df.columns:
            # Get unique values for this factor
            unique_values = set(df[col_a].unique()) | set(df[col_b].unique())
            unique_values.discard(None)
            unique_values = sorted(unique_values)
            
            print(f"\nFactor '{factor_name}': categorical (from nested factors)")
            print(f"  Values: {unique_values}")
            print(f"  Reference: {unique_values[0]}")
            
            # Create difference variable for each value (except reference category)
            for value in unique_values[1:]:  # Skip first as reference category
                diff_col = f'd{factor_name}_{value}'
                df[diff_col] = ((df[col_a] == value).astype(int) - 
                               (df[col_b] == value).astype(int))
                factor_diff_cols.append(diff_col)
    
    # Remove rows with missing values
    required_cols = ['choice']
    required_cols.extend(control_vars)
    required_cols.extend(factor_diff_cols)
    
    df_clean = df[required_cols].dropna()
    print(f"\nAfter removing NAs: {len(df_clean)} rows")
    
    # Fit full model (all variables)
    all_predictors = control_vars + factor_diff_cols
    
    X_full = sm.add_constant(df_clean[all_predictors])
    y = df_clean['choice']
    
    print("\n" + "=" * 60)
    print("FULL MODEL (all variables)")
    print("=" * 60)
    
    full_model = sm.Logit(y, X_full).fit(disp=0)
    print(full_model.summary())
    
    # Fit reduced model (without factor variables)
    reduced_predictors = control_vars  # All configured fields, no factors
    
    X_reduced = sm.add_constant(df_clean[reduced_predictors])
    
    print("\n" + "=" * 60)
    print("REDUCED MODEL (without factors)")
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
                    for name in coef_names if name in factor_diff_cols]
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

