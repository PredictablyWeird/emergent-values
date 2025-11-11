#!/usr/bin/env python3

"""
Analysis script for medical triage experiment results.

Computes and prints average utilities grouped by:
- Severity level
- Group
- Factor (automatically detected from results, e.g., Gender, Race, etc.)
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, Any, Optional
import numpy as np
from scipy import stats


def load_results(results_path: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        results_path: Path to results JSON file
    
    Returns:
        Dictionary containing results data
    """
    with open(results_path, 'r') as f:
        return json.load(f)


def compute_ci(data: list, confidence: float = 0.90) -> tuple:
    """
    Compute confidence interval for a list of values.
    
    Args:
        data: List of numeric values
        confidence: Confidence level (default: 0.90 for 90% CI)
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (0.0, 0.0, 0.0)
    if len(data) == 1:
        return (data[0], data[0], data[0])
    
    data_array = np.array(data)
    mean = np.mean(data_array)
    sem = stats.sem(data_array)  # Standard error of the mean
    h = sem * stats.t.ppf((1 + confidence) / 2, len(data_array) - 1)  # Margin of error
    
    return (mean, mean - h, mean + h)


def analyze_results(results: Dict[str, Any], canonical_factor_value: Optional[str] = None) -> None:
    """
    Analyze results and print average utilities by severity, group, and factor.
    
    Args:
        results: Results dictionary with 'options' and 'utilities'
        canonical_factor_value: Optional reference factor value for comparison
    """
    options = results['options']
    utilities = results['utilities']
    
    # Detect which factor is being used (if any)
    factor_name = None
    for option in options:
        factors = option.get('factors', {})
        if factors:
            # Get the first (and should be only) factor key
            factor_name = list(factors.keys())[0]
            break
    
    # Group utilities by severity, group, and factor
    by_severity = defaultdict(list)
    by_group = defaultdict(list)
    by_factor = defaultdict(list)
    
    for option in options:
        option_id = option['id']
        # Convert to string since utilities dictionary uses string keys
        option_id_str = str(option_id)
        if option_id_str not in utilities:
            continue
        
        utility_mean = utilities[option_id_str]['mean']
        patient_data = option.get('patient_data', {})
        factors = option.get('factors', {})
        
        # Get severity
        severity = patient_data.get('severity')
        if severity is not None:
            by_severity[severity].append(utility_mean)
        
        # Get group
        group = patient_data.get('group')
        if group is not None:
            by_group[group].append(utility_mean)
        
        # Get factor value if factor is present
        if factor_name and factor_name in factors:
            factor_value = factors[factor_name]
            by_factor[factor_value].append(utility_mean)
    
    # Calculate and print averages
    print("=" * 60)
    print("Average Utilities by Severity Level")
    print("=" * 60)
    for severity in sorted(by_severity.keys()):
        utils = by_severity[severity]
        avg = sum(utils) / len(utils) if utils else 0.0
        print(f"Severity {severity}: {avg:.4f} (n={len(utils)})")
    
    print("\n" + "=" * 60)
    print("Average Utilities by Group")
    print("=" * 60)
    for group in sorted(by_group.keys()):
        utils = by_group[group]
        avg = sum(utils) / len(utils) if utils else 0.0
        print(f"Group {group}: {avg:.4f} (n={len(utils)})")
    
    if factor_name and by_factor:
        print("\n" + "=" * 60)
        print(f"Average Utilities by {factor_name}")
        print("=" * 60)
        for factor_value in sorted(by_factor.keys()):
            utils = by_factor[factor_value]
            avg = sum(utils) / len(utils) if utils else 0.0
            print(f"{factor_value}: {avg:.4f} (n={len(utils)})")
    
    # Analysis: Group differences relative to reference group by severity and factor
    # Group utilities by severity, factor, and group
    by_severity_factor_group = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for option in options:
        option_id_str = str(option['id'])
        if option_id_str not in utilities:
            continue
        
        utility_mean = utilities[option_id_str]['mean']
        patient_data = option.get('patient_data', {})
        factors = option.get('factors', {})
        severity = patient_data.get('severity')
        group = patient_data.get('group')
        
        # Get factor value (if factor exists)
        factor_value = None
        if factor_name and factor_name in factors:
            factor_value = factors[factor_name]
        
        if severity is not None and group is not None:
            # Use factor_value as key, or None if no factor
            by_severity_factor_group[severity][factor_value][group].append(utility_mean)
    
    # Determine reference group (prefer Group 1, otherwise use first available group)
    all_groups = set()
    for severity in by_severity_factor_group.keys():
        for factor_val in by_severity_factor_group[severity].keys():
            all_groups.update(by_severity_factor_group[severity][factor_val].keys())
    
    if not all_groups:
        print("\n" + "=" * 60)
        print("Group Differences (by Severity and Factor)")
        print("=" * 60)
        print("No groups found in data")
    else:
        reference_group = 1 if 1 in all_groups else min(all_groups)
        reference_group_name = f"Group {reference_group}"
        
        print("\n" + "=" * 60)
        print(f"Group Differences Relative to {reference_group_name} (by Severity and Factor)")
        print("=" * 60)
        
        # Compute differences for each group relative to reference group, by severity and factor
        group_differences = defaultdict(list)
        
        for severity in sorted(by_severity_factor_group.keys()):
            for factor_val in sorted(by_severity_factor_group[severity].keys(), key=lambda x: (x is None, x)):
                groups_data = by_severity_factor_group[severity][factor_val]
                if reference_group not in groups_data:
                    continue
                
                ref_group_utils = groups_data[reference_group]
                
                for group in sorted(groups_data.keys()):
                    if group == reference_group:
                        continue
                    
                    group_utils = groups_data[group]
                    
                    for ref_util in ref_group_utils:
                        for g_util in group_utils:
                            group_differences[group].append(g_util - ref_util)
        
        if group_differences:
            for group in sorted(group_differences.keys()):
                diffs = group_differences[group]
                mean_diff, ci_lower, ci_upper = compute_ci(diffs, confidence=0.90)
                print(f"Group {group} vs {reference_group_name}: {mean_diff:.4f} (90% CI: [{ci_lower:.4f}, {ci_upper:.4f}], n={len(diffs)})")
        else:
            print(f"No group differences to compute ({reference_group_name} not found or no other groups)")
    
    # Analysis: Factor differences relative to canonical factor value (if specified)
    if canonical_factor_value and factor_name and by_factor:
        if canonical_factor_value not in by_factor:
            print(f"\nWarning: Canonical factor value '{canonical_factor_value}' not found in results.")
            print(f"Available factor values: {sorted(by_factor.keys())}")
        else:
            print("\n" + "=" * 60)
            print(f"Factor Differences Relative to {canonical_factor_value} (by Severity and Group)")
            print("=" * 60)
            
            # Group utilities by severity, group, and factor value
            by_severity_group_factor = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            for option in options:
                option_id_str = str(option['id'])
                if option_id_str not in utilities:
                    continue
                
                utility_mean = utilities[option_id_str]['mean']
                patient_data = option.get('patient_data', {})
                factors = option.get('factors', {})
                severity = patient_data.get('severity')
                group = patient_data.get('group')
                factor_value = factors.get(factor_name) if factor_name else None
                
                if severity is not None and group is not None and factor_value is not None:
                    by_severity_group_factor[severity][group][factor_value].append(utility_mean)
            
            # Compute differences for each factor value relative to canonical value
            factor_differences = defaultdict(list)
            
            for severity in sorted(by_severity_group_factor.keys()):
                for group in sorted(by_severity_group_factor[severity].keys()):
                    factors_data = by_severity_group_factor[severity][group]
                    if canonical_factor_value not in factors_data:
                        continue
                    
                    canonical_utils = factors_data[canonical_factor_value]
                    
                    for factor_value in sorted(factors_data.keys()):
                        if factor_value == canonical_factor_value:
                            continue
                        
                        factor_utils = factors_data[factor_value]
                        
                        for canon_util in canonical_utils:
                            for fact_util in factor_utils:
                                factor_differences[factor_value].append(fact_util - canon_util)
            
            if factor_differences:
                for factor_value in sorted(factor_differences.keys()):
                    diffs = factor_differences[factor_value]
                    mean_diff, ci_lower, ci_upper = compute_ci(diffs, confidence=0.90)
                    print(f"{factor_value} vs {canonical_factor_value}: {mean_diff:.4f} (90% CI: [{ci_lower:.4f}, {ci_upper:.4f}], n={len(diffs)})")
            else:
                print(f"No factor differences to compute (canonical value '{canonical_factor_value}' not found or no other values)")
    
    print()


def analyze_stated_preferences_results(results: Dict[str, Any]) -> None:
    """
    Analyze stated preferences results and print average utilities by factor value and N.
    
    This function works with the stated preferences format where options have:
    - 'X': factor value (e.g., 'male', 'female')
    - 'N': number of patients
    - 'factor': factor name (e.g., 'gender')
    
    Args:
        results: Results dictionary with 'options' and 'utilities'
    """
    options = results['options']
    utilities = results['utilities']
    
    # Group utilities by factor value and by N
    by_factor_value = defaultdict(list)
    by_N = defaultdict(list)
    by_N_and_factor = defaultdict(lambda: defaultdict(list))
    
    for option in options:
        option_id = option['id']
        option_id_str = str(option_id)
        if option_id_str not in utilities:
            continue
        
        utility_mean = utilities[option_id_str]['mean']
        factor_value = option.get('X')  # Factor value (e.g., 'male', 'female')
        N = option.get('N')  # Number of patients
        
        if factor_value is not None:
            by_factor_value[factor_value].append(utility_mean)
        
        if N is not None:
            by_N[N].append(utility_mean)
        
        if N is not None and factor_value is not None:
            by_N_and_factor[N][factor_value].append(utility_mean)
    
    # Calculate and print averages
    if by_factor_value:
        print("=" * 60)
        factor_name = options[0].get('factor', 'Factor') if options else 'Factor'
        print(f"Average Utilities by {factor_name}")
        print("=" * 60)
        for factor_value in sorted(by_factor_value.keys()):
            utils = by_factor_value[factor_value]
            avg = sum(utils) / len(utils) if utils else 0.0
            print(f"{factor_value}: {avg:.4f} (n={len(utils)})")
    
    if by_N:
        print("\n" + "=" * 60)
        print("Average Utilities by Number of Patients (N)")
        print("=" * 60)
        for N in sorted(by_N.keys()):
            utils = by_N[N]
            avg = sum(utils) / len(utils) if utils else 0.0
            print(f"N={N}: {avg:.4f} (n={len(utils)})")
    
    # Analysis: Factor value differences with confidence intervals
    if len(by_factor_value) > 1:
        print("\n" + "=" * 60)
        reference_value = sorted(by_factor_value.keys())[0]
        factor_name = options[0].get('factor', 'Factor') if options else 'Factor'
        print(f"Factor Differences Relative to {reference_value} (by N)")
        print("=" * 60)
        
        # Compute differences controlling for N
        factor_differences = defaultdict(list)
        
        for N in sorted(by_N_and_factor.keys()):
            factor_data = by_N_and_factor[N]
            if reference_value not in factor_data:
                continue
            
            ref_utils = factor_data[reference_value]
            
            for factor_value in sorted(factor_data.keys()):
                if factor_value == reference_value:
                    continue
                
                fv_utils = factor_data[factor_value]
                
                for ref_util in ref_utils:
                    for fv_util in fv_utils:
                        factor_differences[factor_value].append(fv_util - ref_util)
        
        if factor_differences:
            for factor_value in sorted(factor_differences.keys()):
                diffs = factor_differences[factor_value]
                mean_diff, ci_lower, ci_upper = compute_ci(diffs, confidence=0.90)
                print(f"{factor_value} vs {reference_value}: {mean_diff:.4f} (90% CI: [{ci_lower:.4f}, {ci_upper:.4f}], n={len(diffs)})")
        else:
            print(f"No factor differences to compute ({reference_value} not found or no other values)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze medical triage experiment results and print average utilities by severity, group, and factor (automatically detected)."
    )
    parser.add_argument(
        "results_file",
        help="Path to results JSON file (e.g., results_utilities_*.json)"
    )
    parser.add_argument(
        "--canonical_factor_value",
        default=None,
        help="Canonical factor value to use as reference for factor difference analysis (e.g., 'male' for Gender)"
    )
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    
    analyze_results(results, canonical_factor_value=args.canonical_factor_value)


if __name__ == "__main__":
    main()

