"""
Choices: A simple framework for preference elicitation experiments.

Provides the Experiment class for defining and running experiments.
"""

__version__ = "0.2.0"

from .experiment import (
    Experiment,
    ExperimentConfig,
    PromptConfig,
    ResponseData,
)

from .variable import (
    Variable,
    VariableType,
    categorical,
    numerical,
    ordinal,
)

__all__ = [
    'Experiment',
    'ExperimentConfig',
    'PromptConfig',
    'ResponseData',
    'Variable',
    'VariableType',
    'categorical',
    'numerical',
    'ordinal',
]
