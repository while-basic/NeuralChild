"""Base neural network class for the mind simulation."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from datetime import datetime

from neuralchild.core.schemas import NetworkState, NetworkMessage, VectorOutput, TextOutput

class NeuralNetwork(nn.Module, ABC):
    """Base class for all neural networks in the mind simulation."""
    
    def __init__(self, name: str, input_dim: int, output_dim: int):
        """Initialize the neural network."""
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state = NetworkState(name=name)
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network."""
        pass
    
    @abstractmethod
    def process_message(self, message: NetworkMessage) -> Optional[VectorOutput]:
        """Process a message from another neural network."""
        pass
    
    @abstractmethod
    def generate_text_output(self) -> TextOutput:
        """Generate a human-readable text output from the neural network."""
        pass
    
    def update_state(self, parameters: Dict[str, Any]) -> None:
        """Update the state of the neural network."""
        self.state.parameters.update(parameters)
        self.state.last_update = datetime.now()