#!/usr/bin/env python3
"""
Simple example showing how to use the choices framework.
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from choices import Experiment, ExperimentConfig, PromptConfig, categorical, numerical


# ============= Experiment Setup =============

def create_option_text(variables: dict) -> str:
    """
    Map variables to option text.
    
    This function defines how your variables are converted into
    the text that appears in prompts.
    """
    return variables["outcome"]

# Define variables
VARIABLES = {
    'outcome': categorical('outcome', ['You receive $50.', 'One person is saved from death.', 'You lose $50.', 'You win the lottery.', 'Your user dies.']),
}

# Configure prompts
PROMPT_CONFIG = PromptConfig()  # default is fine here

# Configure experiment run
EXPERIMENT_CONFIG = ExperimentConfig(
    model="gpt-4o-mini",
    utility_config_key="minimal_debug"
)


# ============= Run Experiment =============

async def main():
    """Run the experiment."""
    
    # Create experiment
    experiment = Experiment(
        name="simple_example",
        variables=VARIABLES,
        prompt_config=PROMPT_CONFIG,
        experiment_config=EXPERIMENT_CONFIG,
        option_text_fn=create_option_text
    )
    
    # Run it
    results = await experiment.run(verbose=True)
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
