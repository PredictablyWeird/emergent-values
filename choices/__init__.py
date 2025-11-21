"""
Choices: A simple framework for preference elicitation experiments.

Provides the Experiment class for defining and running experiments.
"""

__version__ = "0.2.0"

from .experiment import (
    Experiment,
    ExperimentConfig,
    PromptConfig,
)

from .variable import (
    Variable,
    VariableType,
    categorical,
    numerical,
    log_numerical,
    ordinal,
)

from .results import (
    ExperimentResults,
    PreferenceGraphResults,
    UtilityModelResults,
    ExperimentOption,
)

__all__ = [
    'Experiment',
    'ExperimentConfig',
    'PromptConfig',
    'Variable',
    'VariableType',
    'categorical',
    'numerical',
    'log_numerical',
    'ordinal',
    'ExperimentResults',
    'PreferenceGraphResults',
    'UtilityModelResults',
    'ExperimentOption',
]
