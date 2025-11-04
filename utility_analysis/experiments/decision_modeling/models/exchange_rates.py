"""Exchange rate-based model with cross-validation."""

import math
import pathlib
import numpy as np
from typing import Dict, List, Tuple
import statsmodels.api as sm

from predictors.log_utility import compute_log_utility_predictions
from predictors.exchange_rates import compute_exchange_rate_predictions


# Import two_way_geometric_exchange_rate from create_exchange_rates_plots
def _get_two_way_geometric_exchange_rate():
    """Get two_way_geometric_exchange_rate function from create_exchange_rates_plots."""
    import sys
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(base_dir))
    
    try:
        from create_exchange_rates_plots import two_way_geometric_exchange_rate
        return two_way_geometric_exchange_rate
    except ImportError:
        return None


two_way_geometric_exchange_rate = _get_two_way_geometric_exchange_rate()
if two_way_geometric_exchange_rate is None:
    raise ImportError("Could not import two_way_geometric_exchange_rate from create_exchange_rates_plots")


def fit_utility_curves_from_data(
    metadata: List[Dict[str, str]],
    N_a: np.ndarray,
    N_b: np.ndarray,
    y: np.ndarray,
    scale: float = 1.0,
    method: str = "normal"
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Fit utility curves from decision modeling data using Thurstonian model approach.
    
    This matches the approach in compute_utilities:
    1. Fit Thurstonian probit model to get utilities for each (country, N) option
    2. Fit linear regression: utility = intercept + slope * ln(N) for each country
    
    Utility curves are of the form: utility = intercept + slope * ln(N)
    
    Args:
        metadata: List of dictionaries with 'country_a' and 'country_b' keys
        N_a: Array of N_a values
        N_b: Array of N_b values
        y: Array of probabilities (probability that country A is preferred)
        scale: Not used (kept for API compatibility, Thurstonian model handles scale internally)
        method: "normal" (probit) or "sigmoid" (logistic) - only "normal" is supported for Thurstonian
        
    Returns:
        Tuple of (slopes, intercepts) dictionaries mapping country names to their slope/intercept values
    """
    import sys
    # Add utility_analysis directory to path
    utility_analysis_dir = pathlib.Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(utility_analysis_dir))
    
    # Import Thurstonian model components
    from compute_utilities.compute_utilities import PreferenceGraph, PreferenceEdge
    from compute_utilities.utility_models.thurstonian.utils import fit_thurstonian_model
    
    if method != "normal":
        raise ValueError(f"Thurstonian model only supports 'normal' (probit) method, got '{method}'")
    
    # Create unique options for each (country, N) combination
    option_id_to_option = {}
    option_counter = 0
    
    # First pass: collect all unique (country, N) combinations
    for i, meta in enumerate(metadata):
        country_a = meta['country_a']
        country_b = meta['country_b']
        n_a = float(N_a[i])
        n_b = float(N_b[i])
        
        # Create option IDs
        option_a_key = (country_a, n_a)
        option_b_key = (country_b, n_b)
        
        if option_a_key not in option_id_to_option:
            option_id = option_counter
            option_counter += 1
            option_id_to_option[option_a_key] = {
                'id': option_id,
                'description': f"{country_a} with N={n_a}",
                'country': country_a,
                'N': n_a
            }
        
        if option_b_key not in option_id_to_option:
            option_id = option_counter
            option_counter += 1
            option_id_to_option[option_b_key] = {
                'id': option_id,
                'description': f"{country_b} with N={n_b}",
                'country': country_b,
                'N': n_b
            }
    
    # Create options list
    options = list(option_id_to_option.values())
    
    if len(options) < 2:
        return {}, {}
    
    # Create PreferenceGraph
    graph = PreferenceGraph(options=options, holdout_fraction=0.0, seed=42)
    
    # Add edges with probabilities
    for i, meta in enumerate(metadata):
        country_a = meta['country_a']
        country_b = meta['country_b']
        n_a = float(N_a[i])
        n_b = float(N_b[i])
        prob = float(y[i])
        
        # Clamp probability to avoid infinity
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        
        option_a_key = (country_a, n_a)
        option_b_key = (country_b, n_b)
        
        option_A = option_id_to_option[option_a_key]
        option_B = option_id_to_option[option_b_key]
        
        # Create edge
        edge = PreferenceEdge(
            option_A=option_A,
            option_B=option_B,
            probability_A=prob,
            aux_data={}
        )
        
        edge_index = (option_A['id'], option_B['id'])
        graph.edges[edge_index] = edge
        # Also add to training_edges set for consistency
        if edge_index in graph.training_edges_pool:
            graph.training_edges_pool.remove(edge_index)
            graph.training_edges.add(edge_index)
    
    # Fit Thurstonian model
    try:
        utilities, model_log_loss, model_accuracy = fit_thurstonian_model(
            graph=graph,
            num_epochs=1000,
            learning_rate=0.01
        )
    except Exception as e:
        print(f"Error fitting Thurstonian model: {e}")
        return {}, {}
    
    # Now fit curves: utility = intercept + slope * ln(N) for each country
    # Group utilities by country
    country_utilities = {}  # country -> list of (lnN, utility) tuples
    
    for option_key, option_data in option_id_to_option.items():
        country = option_data['country']
        n_val = option_data['N']
        option_id = option_data['id']
        
        if option_id not in utilities:
            continue
        
        utility_mean = utilities[option_id]['mean']
        
        if n_val > 0:
            lnN = math.log(n_val)
            if country not in country_utilities:
                country_utilities[country] = []
            country_utilities[country].append((lnN, utility_mean))
    
    # Fit linear regression for each country
    slopes = {}
    intercepts = {}
    
    for country, data_points in country_utilities.items():
        if len(data_points) < 2:
            # Need at least 2 points to fit a line
            continue
        
        lnN_vals = np.array([x[0] for x in data_points])
        utility_vals = np.array([x[1] for x in data_points])
        
        # Fit: utility = intercept + slope * ln(N)
        X_design = sm.add_constant(lnN_vals)
        model = sm.OLS(utility_vals, X_design).fit()
        
        intercepts[country] = model.params[0]
        slopes[country] = model.params[1]
    
    return slopes, intercepts


def estimate_exchange_rates_from_utility_curves(
    slopes: Dict[str, float],
    intercepts: Dict[str, float],
    N_values: List[float] | None = None
) -> Dict[Tuple[str, str], float]:
    """
    Estimate exchange rates from utility curves.
    
    Args:
        slopes: Dictionary mapping country names to slopes
        intercepts: Dictionary mapping country names to intercepts
        N_values: List of N values to use for exchange rate calculation (default: use common N values)
        
    Returns:
        Dictionary mapping (country_a, country_b) tuples to exchange rates
    """
    if N_values is None:
        # Use common N values from 1 to 1000
        N_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    
    exchange_rates = {}
    countries = list(slopes.keys())
    
    for country_a in countries:
        for country_b in countries:
            if country_a == country_b:
                continue
            
            try:
                exchange_rate = two_way_geometric_exchange_rate(
                    country_a, country_b, N_values,
                    slopes, intercepts,
                    skip_if_negative_slope=False,
                    allow_negative_slopes=True
                )
                
                if exchange_rate is not None and not math.isinf(exchange_rate) and exchange_rate > 0:
                    exchange_rates[(country_a, country_b)] = exchange_rate
            except (ZeroDivisionError, ValueError, OverflowError):
                # Skip this exchange rate if calculation fails
                continue
    
    return exchange_rates



