"""
Variable definitions for experiments.

Variables define the dimensions along which options vary.
"""

from dataclasses import dataclass
from typing import List, Any, Literal
from enum import Enum


class VariableType(Enum):
    """Type of variable."""
    CATEGORICAL = "categorical"  # Discrete categories (gender, country, etc.)
    NUMERICAL = "numerical"      # Numeric values (N, age, etc.)
    ORDINAL = "ordinal"          # Ordered categories (age groups, etc.)


@dataclass
class Variable:
    """
    A variable that can take different values in an experiment.
    
    Attributes:
        name: Variable name (e.g., 'gender', 'N', 'country')
        values: List of possible values this variable can take
        type: Type of variable (categorical, numerical, or ordinal)
        description: Optional human-readable description
    """
    name: str
    values: List[Any]
    type: VariableType = VariableType.CATEGORICAL
    description: str = ""
    
    def __post_init__(self):
        """Validate variable configuration."""
        if not self.values:
            raise ValueError(f"Variable '{self.name}' must have at least one value")
        
        # Auto-detect numerical type if all values are numbers
        if self.type == VariableType.CATEGORICAL and self.values:
            if all(isinstance(v, (int, float)) for v in self.values):
                # Could warn or auto-convert, but let's be explicit
                pass
    
    def __len__(self) -> int:
        """Return number of possible values."""
        return len(self.values)
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'values': self.values,
            'type': self.type.value,
            'description': self.description
        }


def categorical(name: str, values: List[Any], description: str = "") -> Variable:
    """Helper to create a categorical variable."""
    return Variable(name=name, values=values, type=VariableType.CATEGORICAL, description=description)


def numerical(name: str, values: List[float], description: str = "") -> Variable:
    """Helper to create a numerical variable."""
    return Variable(name=name, values=values, type=VariableType.NUMERICAL, description=description)


def ordinal(name: str, values: List[Any], description: str = "") -> Variable:
    """Helper to create an ordinal variable."""
    return Variable(name=name, values=values, type=VariableType.ORDINAL, description=description)

