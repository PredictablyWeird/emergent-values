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
from typing import Dict, Any


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


def analyze_results(results: Dict[str, Any]) -> None:
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
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze medical triage experiment results and print average utilities by severity, group, and factor (automatically detected)."
    )
    parser.add_argument(
        "results_file",
        help="Path to results JSON file (e.g., results_utilities_*.json)"
    )
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    
    analyze_results(results)


if __name__ == "__main__":
    main()

