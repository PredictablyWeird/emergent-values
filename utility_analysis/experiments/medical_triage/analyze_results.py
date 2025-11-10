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
    
    # Analysis: Group differences relative to Group 1 by severity
    print("\n" + "=" * 60)
    print("Group Differences Relative to Group 1 (by Severity)")
    print("=" * 60)
    
    # Group utilities by severity and group
    by_severity_group = defaultdict(lambda: defaultdict(list))
    for option in options:
        option_id_str = str(option['id'])
        if option_id_str not in utilities:
            continue
        
        utility_mean = utilities[option_id_str]['mean']
        patient_data = option.get('patient_data', {})
        severity = patient_data.get('severity')
        group = patient_data.get('group')
        
        if severity is not None and group is not None:
            by_severity_group[severity][group].append(utility_mean)
    
    # Compute differences for each group relative to Group 1, by severity
    group_differences = defaultdict(list)  # group -> list of differences
    
    for severity in sorted(by_severity_group.keys()):
        groups_data = by_severity_group[severity]
        if 1 not in groups_data:
            continue  # Skip if group 1 doesn't exist for this severity
        
        group1_utils = groups_data[1]
        
        for group in sorted(groups_data.keys()):
            if group == 1:
                continue  # Skip group 1 itself
            
            group_utils = groups_data[group]
            
            # Compute differences for each pair within this severity level
            for g1_util in group1_utils:
                for g_util in group_utils:
                    diff = g_util - g1_util
                    group_differences[group].append(diff)
    
    if group_differences:
        for group in sorted(group_differences.keys()):
            diffs = group_differences[group]
            mean_diff, ci_lower, ci_upper = compute_ci(diffs, confidence=0.90)
            print(f"Group {group} vs Group 1: {mean_diff:.4f} (90% CI: [{ci_lower:.4f}, {ci_upper:.4f}], n={len(diffs)})")
    else:
        print("No group differences to compute (Group 1 not found or no other groups)")
    
    # Analysis: Factor differences relative to canonical factor value (if specified)
    if canonical_factor_value and factor_name and by_factor:
        if canonical_factor_value not in by_factor:
            print(f"\nWarning: Canonical factor value '{canonical_factor_value}' not found in results.")
            print(f"Available factor values: {sorted(by_factor.keys())}")
        else:
            print("\n" + "=" * 60)
            print(f"Factor Differences Relative to {canonical_factor_value} (by Severity)")
            print("=" * 60)
            
            # Group utilities by severity and factor value
            by_severity_factor = defaultdict(lambda: defaultdict(list))
            for option in options:
                option_id_str = str(option['id'])
                if option_id_str not in utilities:
                    continue
                
                utility_mean = utilities[option_id_str]['mean']
                patient_data = option.get('patient_data', {})
                factors = option.get('factors', {})
                severity = patient_data.get('severity')
                factor_value = factors.get(factor_name) if factor_name else None
                
                if severity is not None and factor_value is not None:
                    by_severity_factor[severity][factor_value].append(utility_mean)
            
            # Compute differences for each factor value relative to canonical value, by severity
            factor_differences = defaultdict(list)  # factor_value -> list of differences
            
            for severity in sorted(by_severity_factor.keys()):
                factors_data = by_severity_factor[severity]
                if canonical_factor_value not in factors_data:
                    continue  # Skip if canonical factor value doesn't exist for this severity
                
                canonical_utils = factors_data[canonical_factor_value]
                
                for factor_value in sorted(factors_data.keys()):
                    if factor_value == canonical_factor_value:
                        continue  # Skip canonical factor value itself
                    
                    factor_utils = factors_data[factor_value]
                    
                    # Compute differences for each pair within this severity level
                    for canon_util in canonical_utils:
                        for fact_util in factor_utils:
                            diff = fact_util - canon_util
                            factor_differences[factor_value].append(diff)
            
            if factor_differences:
                for factor_value in sorted(factor_differences.keys()):
                    diffs = factor_differences[factor_value]
                    mean_diff, ci_lower, ci_upper = compute_ci(diffs, confidence=0.90)
                    print(f"{factor_value} vs {canonical_factor_value}: {mean_diff:.4f} (90% CI: [{ci_lower:.4f}, {ci_upper:.4f}], n={len(diffs)})")
            else:
                print(f"No factor differences to compute (canonical value '{canonical_factor_value}' not found or no other values)")
    
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

