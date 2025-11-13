"""
Core module for the choices experiment framework.

Provides the Experiment class and configuration classes for running experiments.
"""

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
