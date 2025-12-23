"""Functions for loading country features and decision data."""

import csv
import json
import os
from typing import Dict, List, Tuple


def load_country_features(jsonl_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load country features from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file containing country features
        
    Returns:
        Dictionary mapping country names to their feature dictionaries.
    """
    country_features = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            country_name = record['country']
            features = record['features']
            country_features[country_name] = features
    return country_features


def create_decision_file(results_dir: str, filepath: str, model_name: str) -> str:
    """
    Create decision file from exchange rate results.
    
    Args:
        results_dir: Directory containing exchange rate results
        filepath: Path where the CSV file should be created
        model_name: Model name for loading results
        
    Returns:
        Path to the created CSV file
    """
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(
            f"'{results_dir}' doesn't exist.\n"
            f"Expected {results_dir}."
        )
    
    results_file = os.path.join(results_dir, f'results_{model_name}.json')
    responses = json.load(open(results_file))['graph_data']['edges']

    # Open the file in write mode ('w') with newline='' to prevent extra blank rows
    probs = []
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        for k, v in responses.items():
            countA = v['option_A']['description'].split()[0]
            countryA = " ".join(v['option_A']['description'].split()[3:-5])
            countB = v['option_B']['description'].split()[0]
            countryB = " ".join(v['option_B']['description'].split()[3:-5])
            aux_data = v['aux_data']
            if 'is_pseudolabel' in aux_data and aux_data['is_pseudolabel'] == True:
                continue
            probA = float(v['probability_A'])
            if countA == 'You' or countB == 'You':
                continue
            countA, countB = int(countA), int(countB)
            writer.writerow([int(countA), countryA, countB, countryB, probA])
            probs.append((countA, countB, probA))
    return filepath


def load_decision_file(
    model_name: str,
    csv_path: str | None = None,
    category: str = "countries",
    measure: str = "terminal_illness"
) -> List[Tuple[float, str, float, str, float]]:
    """
    Load decision file (CSV) with country comparisons.
    
    Args:
        model_name: Model name
        csv_path: Optional path to CSV file. If None, will construct from model_name
        category: Category (e.g., 'countries')
        measure: Measure (e.g., 'terminal_illness')
    
    Returns:
        List of tuples: (N_a, country_a, N_b, country_b, probability)
    """
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'exchange_rates', 'results', model_name, category, measure
    )
    results_dir = os.path.normpath(results_dir)
    
    if csv_path is None:
        csv_path = os.path.join(results_dir, f'decisions_{model_name}.csv')

    if not os.path.exists(csv_path):
        print(f"Decision file not found for model: {model_name}. Creating new decision file...")
        csv_path = create_decision_file(results_dir=results_dir, filepath=csv_path, model_name=model_name)

    decisions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # CSV format: N_a, country_a, N_b, country_b, probability
            if len(row) >= 5:
                N_a = float(row[0])
                country_a = row[1]
                N_b = float(row[2])
                country_b = row[3]
                probability = float(row[4])
                decisions.append((N_a, country_a, N_b, country_b, probability))
    return decisions

