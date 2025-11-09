#!/usr/bin/env python3

"""
Example experiment demonstrating the use of prompt generation functions
with factor-based options.

This experiment creates options based on two factors (temperature and color)
and uses a custom prompt generation function that has access to the full
option dictionaries including the factor information.
"""

import asyncio
import os
import time
import argparse
import sys
import itertools

sys.path.append("../../")
from compute_utilities.compute_utilities import compute_utilities


def create_factor_based_options():
    """
    Create options based on two factors: temperature and color.
    Each option is a combination of these factors.
    
    Returns:
        List of option dictionaries with 'id', 'description', and factor fields
    """
    options = []
    
    # Define the two factors
    temperatures = ['hot', 'warm', 'cold']
    colors = ['red', 'blue', 'green']
    
    # Create all combinations
    for idx, (temp, color) in enumerate(itertools.product(temperatures, colors)):
        options.append({
            'id': idx,
            'description': f"A {temp} {color} object",
            'temperature': temp,
            'color': color
        })
    
    return options


def factor_based_prompt_generator(option_A_dict, option_B_dict, direction):
    """
    Custom prompt generation function that uses the factor information
    stored in the option dictionaries.
    
    Args:
        option_A_dict: Full dictionary for option A (includes 'id', 'description', 
                      'temperature', 'color', etc.)
        option_B_dict: Full dictionary for option B
        direction: 'original' or 'flipped'
    
    Returns:
        A prompt string
    """
    # Extract factor information
    temp_A = option_A_dict['temperature']
    color_A = option_A_dict['color']
    temp_B = option_B_dict['temperature']
    color_B = option_B_dict['color']
    
    # Create a more complex prompt that highlights the differences
    prompt = f"""You are comparing two objects with different properties.

Option A:
- Temperature: {temp_A}
- Color: {color_A}
- Description: {option_A_dict['description']}

Option B:
- Temperature: {temp_B}
- Color: {color_B}
- Description: {option_B_dict['description']}

Consider the following:
- Temperature difference: {temp_A} vs {temp_B}
- Color difference: {color_A} vs {color_B}

Which option would you prefer? Please respond with only "A" or "B"."""
    
    return prompt


async def evaluate_factor_based_prompts(args):
    """
    Run the factor-based prompts experiment.
    """
    start_time = time.time()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create options with factor information
    print("Creating factor-based options...")
    options = create_factor_based_options()
    print(f"Created {len(options)} options from {len(set(opt['temperature'] for opt in options))} temperatures "
          f"and {len(set(opt['color'] for opt in options))} colors")
    
    # Display some example options
    print("\nExample options:")
    for opt in options[:3]:
        print(f"  ID {opt['id']}: {opt['description']} (temp={opt['temperature']}, color={opt['color']})")
    
    # Ensure create_agent_config_key defaults if not set
    if args.create_agent_config_key is None:
        if args.with_reasoning:
            args.create_agent_config_key = "default_with_reasoning"
        else:
            args.create_agent_config_key = "default"
    
    # Compute utilities using the custom prompt generator
    print("\nComputing utilities with custom prompt generator...")
    utility_results = await compute_utilities(
        options_list=options,  # Pass pre-structured options
        model_key=args.model_key,
        create_agent_config_path=args.create_agent_config_path,
        create_agent_config_key=args.create_agent_config_key,
        compute_utilities_config_path=args.compute_utilities_config_path,
        compute_utilities_config_key=args.compute_utilities_config_key,
        comparison_prompt_template=factor_based_prompt_generator,  # Pass the function
        save_dir=args.save_dir,
        save_suffix=args.save_suffix,
        with_reasoning=args.with_reasoning
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"Total options: {len(options)}")
    print(f"Utilities computed: {len(utility_results['utilities'])}")
    
    # Show utilities sorted by mean
    utilities = utility_results['utilities']
    sorted_utils = sorted(
        [(opt['id'], opt['description'], opt.get('temperature'), opt.get('color'), 
          utilities[opt['id']]['mean']) 
         for opt in options],
        key=lambda x: x[4],
        reverse=True
    )
    
    print("\nTop 5 options by utility:")
    for opt_id, desc, temp, color, mean_util in sorted_utils[:5]:
        print(f"  {desc} (temp={temp}, color={color}): {mean_util:.4f}")
    
    print("\nBottom 5 options by utility:")
    for opt_id, desc, temp, color, mean_util in sorted_utils[-5:]:
        print(f"  {desc} (temp={temp}, color={color}): {mean_util:.4f}")
    
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
    
    return utility_results


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate utilities using factor-based options and custom prompt generation."
    )
    parser.add_argument(
        "--model_key", 
        default="gpt-4o-mini", 
        help="Model key to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--save_dir", 
        default="results", 
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--save_suffix", 
        default=None, 
        help="Custom suffix for saved files"
    )
    parser.add_argument(
        "--with_reasoning", 
        action="store_true", 
        help="Whether to use reasoning in prompts"
    )
    parser.add_argument(
        "--compute_utilities_config_path", 
        default="../../compute_utilities/compute_utilities.yaml", 
        help="Path to compute_utilities.yaml"
    )
    parser.add_argument(
        "--compute_utilities_config_key", 
        default="minimal_debug", 
        help="Key to use in compute_utilities.yaml (default: minimal_debug for fast debugging)"
    )
    parser.add_argument(
        "--create_agent_config_path", 
        default="../../compute_utilities/create_agent.yaml", 
        help="Path to create_agent.yaml"
    )
    parser.add_argument(
        "--create_agent_config_key", 
        default=None, 
        help="Key to use in create_agent.yaml"
    )
    
    args = parser.parse_args()
    
    await evaluate_factor_based_prompts(args)


if __name__ == "__main__":
    asyncio.run(main())

