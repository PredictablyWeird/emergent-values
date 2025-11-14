"""
Experiment class for defining and running preference elicitation experiments.

Uses the existing PreferenceGraph from compute_utilities.
"""

import os
import sys
import itertools
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime

# Add utility_analysis to path
sys.path.append(str(Path(__file__).parent.parent.parent / "utility_analysis"))

from compute_utilities.compute_utilities import PreferenceGraph, compute_utilities
from .variable import Variable, VariableType


@dataclass
class ExperimentConfig:
    """
    Configuration for running an experiment.
    
    Attributes:
        model: Model key (e.g., 'gpt-4o-mini')
        utility_config_path: Path to compute_utilities config
        utility_config_key: Key in compute_utilities config
        agent_config_path: Path to create_agent config
        agent_config_key: Key in create_agent config (None = auto-select)
    """
    model: str = "gpt-4o-mini"
    utility_config_path: str = "utility_analysis/compute_utilities/compute_utilities.yaml"
    utility_config_key: str = "thurstonian_active_learning_k5"
    agent_config_path: str = "utility_analysis/compute_utilities/create_agent.yaml"
    agent_config_key: Optional[str] = None


@dataclass
class PromptConfig:
    """
    Configuration for prompt generation.
    
    The prompt is built from three components:
    1. setup - The question/context (e.g., "Which option do you prefer?")
    2. option_list - How options are presented with {option_A} and {option_B} placeholders
    3. instructions - Response instructions (e.g., "Please respond with only 'A' or 'B'.")
    
    Attributes:
        system_prompt: System message for the LLM
        setup: Question or context for the comparison
        option_list: Format for presenting options (must include {option_A} and {option_B})
        instructions: Instructions for how to respond
        with_reasoning: Whether to use reasoning
        fixed_variables: Variables that don't vary within run but are compared across runs
    """
    system_prompt: str = "You are a helpful assistant."
    setup: str = "The following two options describe observations about the state of the world. Which implied state of the world would you prefer?"
    option_list: str = "Option A:\n{option_A}\n\nOption B:\n{option_B}"
    instructions: str = "Please respond with only \"A\" or \"B\"."
    with_reasoning: bool = False
    fixed_variables: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def template(self) -> str:
        """Dynamically generate the full template from components."""
        parts = []
        if self.setup:
            parts.append(self.setup)
        if self.option_list:
            parts.append(self.option_list)
        if self.instructions:
            parts.append(self.instructions)
        return "\n\n".join(parts)
#TODO Check if this actually works with reasoning. (Anyway need to revise to also store the reasoning)


@dataclass
class ResponseData:
    """
    Parsed response data from an experiment run.
    
    Customize this based on what you want to extract from responses.
    """
    raw_response: str
    parsed_data: Dict[str, Any] = field(default_factory=dict)


class Experiment:
    """
    An experiment defines how to run preference elicitation.
    
    Core responsibilities:
    - Define input variables and their values
    - Generate options (nodes) from variable combinations
    - Map variables to option text for prompts
    - Optional: Filter which edges to exclude
    - Parse responses and extract data
    """
    
    def __init__(
        self,
        name: str,
        variables: Union[Dict[str, List[Any]], Dict[str, Variable]],
        prompt_config: PromptConfig,
        experiment_config: ExperimentConfig,
        option_text_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
        edge_filter: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
        response_parser: Optional[Callable[[str], ResponseData]] = None,
        run_id: Optional[str] = None,
        unique_fields: Optional[List[str]] = None,
        custom_prompt_generator: Optional[Callable[[Dict, Dict, str], str]] = None
    ):
        """
        Initialize experiment.
        
        Args:
            name: Experiment name (used for directories)
            variables: Dict mapping variable names to lists of values OR Variable objects
            prompt_config: Configuration for prompt generation (can have empty fields if using custom_prompt_generator)
            experiment_config: Configuration for running the experiment
            option_text_fn: Function to generate text from variables dict. Not used if custom_prompt_generator is provided.
            edge_filter: Optional function returning False to exclude edge.
                        Called with (variables_a, variables_b)
            response_parser: Function to parse LLM responses. If None, uses default
            run_id: Run identifier. If None, auto-generated
            unique_fields: Fields passed to PreferenceGraph for edge filtering
            custom_prompt_generator: Optional function(option_A_dict, option_B_dict, direction) -> prompt_str
                                    If provided, this overrides the PromptConfig template-based generation
        """
        self.name = self._sanitize_name(name)
        
        # Normalize variables to Variable objects
        self.variables = self._normalize_variables(variables)
        
        self.prompt_config = prompt_config
        self.experiment_config = experiment_config
        self.option_text_fn = option_text_fn
        self.edge_filter = edge_filter
        self.response_parser = response_parser or self._default_response_parser
        self.run_id = run_id or self._generate_run_id()
        self.unique_fields = unique_fields
        self.custom_prompt_generator = custom_prompt_generator
        
        # Generated lazily
        self._options = None
        self._preference_graph = None
    
    def _normalize_variables(self, variables: Union[Dict[str, List[Any]], Dict[str, Variable]]) -> Dict[str, Variable]:
        """
        Normalize variables to Variable objects.
        
        Accepts either:
        - Dict[str, List[Any]] - converts to Variable objects
        - Dict[str, Variable] - returns as-is
        """
        normalized = {}
        for key, value in variables.items():
            if isinstance(value, Variable):
                normalized[key] = value
            elif isinstance(value, list):
                # Auto-detect type: numerical if all values are numbers
                if all(isinstance(v, (int, float)) for v in value):
                    var_type = VariableType.NUMERICAL
                else:
                    var_type = VariableType.CATEGORICAL
                normalized[key] = Variable(name=key, values=value, type=var_type)
            else:
                raise ValueError(f"Variable '{key}' must be either a list or Variable object, got {type(value)}")
        return normalized
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for use in directory paths."""
        # Replace spaces and special chars with underscores
        import re
        return re.sub(r'[^\w\-]', '_', name)
    
    def _default_response_parser(self, response: str) -> ResponseData:
        """Default response parser - just stores raw response."""
        return ResponseData(raw_response=response)
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #return f"{self.name}_{self.experiment_config.model}_{timestamp}"
        return timestamp
    
    def _generate_options(self) -> List[Dict[str, Any]]:
        """
        Generate all options from cartesian product of variables.
        
        Returns list of option dicts with:
        - 'id': unique identifier (simple integer)
        - 'description': text representation
        - All variable key-value pairs
        """
        options = []
        var_names = list(self.variables.keys())
        var_values = [self.variables[name].values for name in var_names]
        
        for idx, combo in enumerate(itertools.product(*var_values)):
            variables = dict(zip(var_names, combo))
            
            # Generate text
            if self.option_text_fn:
                text = self.option_text_fn(variables)
            else:
                # If no option_text_fn provided (e.g., when using custom prompt generator),
                # create a simple description from variables
                text = ", ".join([f"{name}={val}" for name, val in variables.items()])
            
            # Generate simple integer ID
            option_id = idx
            
            # Create option dict with all variable data
            option = {
                'id': option_id,
                'description': text,
                **variables  # Include all variables in the option dict
            }
            
            # Extract simple fields from nested dicts for better summary formatting
            # (compute_utilities skips dict/list fields when generating summaries)
            for var_name, var_value in variables.items():
                if isinstance(var_value, dict):
                    # Extract an 'id' or '_id' field if it exists
                    for id_field in ['id', 'patient_id', 'item_id', 'option_id']:
                        if id_field in var_value:
                            option[f'{var_name}_{id_field}'] = var_value[id_field]
                            break
            
            options.append(option)
        
        return options
    
    def get_options(self) -> List[Dict[str, Any]]:
        """Get all options (generates if needed)."""
        if self._options is None:
            self._options = self._generate_options()
        return self._options
    
    def get_preference_graph(self) -> PreferenceGraph:
        """
        Get PreferenceGraph (creates if needed).
        
        Uses edge_filter if provided to exclude certain comparisons.
        """
        if self._preference_graph is None:
            options = self.get_options()
            
            # If we have an edge_filter function, we need to filter the edges
            # PreferenceGraph has unique_fields for built-in filtering
            # For custom filtering, we'd need to subclass or post-filter
            
            self._preference_graph = PreferenceGraph(
                options=options,
                holdout_fraction=0.0,  # No holdout by default
                unique_fields=self.unique_fields
            )
            
            # Apply custom edge filter if provided
            if self.edge_filter:
                # Filter training_edges_pool based on edge_filter
                options_by_id = {opt['id']: opt for opt in options}
                filtered_edges = set()
                
                for (id_a, id_b) in self._preference_graph.training_edges_pool:
                    opt_a = options_by_id[id_a]
                    opt_b = options_by_id[id_b]
                    
                    # Keep edge if filter returns True
                    if self.edge_filter(opt_a, opt_b):
                        filtered_edges.add((id_a, id_b))
                
                self._preference_graph.training_edges_pool = filtered_edges
        
        return self._preference_graph
    
    def parse_response(self, response: str) -> ResponseData:
        """Parse an LLM response."""
        return self.response_parser(response)
    
    def _save_example_prompt(self, graph: PreferenceGraph, save_path: str):
        """
        Save an example prompt from a random edge in the graph.
        
        Args:
            graph: PreferenceGraph with edges
            save_path: Directory to save the example prompt
        """
        import random
        
        # Get a random edge
        if not graph.training_edges_pool:
            return  # No edges to sample from
        
        edge_ids = list(graph.training_edges_pool)
        random_edge = random.choice(edge_ids)
        
        # Get the nodes
        node_a_id, node_b_id = random_edge
        node_a = graph.options_by_id[node_a_id]
        node_b = graph.options_by_id[node_b_id]
        
        # Generate the prompt text
        if self.custom_prompt_generator:
            # Use custom prompt generator - it receives the full option dictionaries
            prompt_text = self.custom_prompt_generator(node_a, node_b, 'original')
        else:
            # Use template-based generation
            option_a_text = node_a['description']
            option_b_text = node_b['description']
            
            prompt_text = self.prompt_config.template.format(
                option_A=option_a_text,
                option_B=option_b_text
            )
        
        # Create full example with system message
        full_example = f"System Message:\n{self.prompt_config.system_prompt}\n\n{'='*60}\n\n{prompt_text}"
        
        # Save to file
        example_path = os.path.join(save_path, "example_prompt.txt")
        with open(example_path, 'w') as f:
            f.write(full_example)
            f.write(f"\n\n{'='*60}\n")
            f.write(f"Option A ID: {node_a_id}\n")
            f.write(f"Option B ID: {node_b_id}\n")
    
    def get_save_dir(self, base_dir: str = "results") -> str:
        """Get directory for saving results."""
        return os.path.join(base_dir, self.name, self.experiment_config.model, self.run_id)
    
    async def run(self, save_dir: str = "results", verbose: bool = True):
        """
        Run the experiment.
        
        Args:
            save_dir: Base directory for results
            verbose: Whether to print progress
            
        Returns:
            Dictionary with results including utilities and metrics
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Running Experiment: {self.name}")
            print(f"Run ID: {self.run_id}")
            print(f"Model: {self.experiment_config.model}")
            print(f"{'='*80}\n")
        
        # Get options and graph
        options = self.get_options()
        graph = self.get_preference_graph()
        
        if verbose:
            print(f"Generated {len(options)} options from variables:")
            for var_name, var in self.variables.items():
                print(f"  {var_name} ({var.type.value}): {len(var)} values")
            print(f"\nGraph edges: {len(graph.training_edges_pool)}")
            print(f"\nExample options:")
            for opt in options[:3]:
                print(f"  - {opt['description']}")
        
        # Determine agent config key
        agent_config_key = self.experiment_config.agent_config_key
        if agent_config_key is None:
            agent_config_key = (
                "default_with_reasoning" if self.prompt_config.with_reasoning 
                else "default"
            )
        
        # Create save directory
        save_path = self.get_save_dir(base_dir=save_dir)
        os.makedirs(save_path, exist_ok=True)
        
        if verbose:
            print(f"\nSave directory: {save_path}")
        
        # Save example prompt from a random edge
        self._save_example_prompt(graph, save_path)
        
        if verbose:
            print(f"Saved example prompt to: {os.path.join(save_path, 'example_prompt.txt')}")
            print(f"\nRunning compute_utilities...")
        
        # Determine which prompt template/generator to use
        if self.custom_prompt_generator:
            comparison_prompt = self.custom_prompt_generator
        else:
            comparison_prompt = self.prompt_config.template
        
        # Run compute_utilities
        results = await compute_utilities(
            options_list=options,
            model_key=self.experiment_config.model,
            create_agent_config_path=self.experiment_config.agent_config_path,
            create_agent_config_key=agent_config_key,
            compute_utilities_config_path=self.experiment_config.utility_config_path,
            compute_utilities_config_key=self.experiment_config.utility_config_key,
            save_dir=save_path,
            save_suffix=None,
            with_reasoning=self.prompt_config.with_reasoning,
            system_message=self.prompt_config.system_prompt,
            comparison_prompt_template=comparison_prompt,
        )
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Experiment complete!")
            if 'utilities' in results:
                print(f"Computed utilities for {len(results['utilities'])} options")
            print(f"Results saved to: {save_path}")
            print(f"{'='*80}\n")
        
        return results
