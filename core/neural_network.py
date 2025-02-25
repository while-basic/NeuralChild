"""Base neural network class for the mind simulation.

This module provides the foundation for all neural networks in the Neural Child project,
implementing core functionality for network development and communication.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np

from core.schemas import NetworkState, NetworkMessage, VectorOutput, TextOutput, DevelopmentalStage

class NeuralNetwork(nn.Module, ABC):
    """Base class for all neural networks in the mind simulation.
    
    This abstract base class provides core functionality for development-aware neural networks
    that can adapt and grow based on the developmental stage of the mind.
    """
    
    def __init__(self, name: str, input_dim: int, output_dim: int):
        """Initialize the neural network.
        
        Args:
            name: Unique identifier for the network
            input_dim: Dimension of input vectors
            output_dim: Dimension of output vectors
        """
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state = NetworkState(
            name=name,
            developmental_weights={
                stage: 0.0 for stage in DevelopmentalStage
            }
        )
        self.developmental_stage = DevelopmentalStage.INFANT
        self.last_activations = []
        self.learning_rate = 0.01
        self.experience_count = 0
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def process_message(self, message: NetworkMessage) -> Optional[VectorOutput]:
        """Process a message from another neural network.
        
        Args:
            message: Message from another network
            
        Returns:
            Optional vector output as response
        """
        pass
    
    @abstractmethod
    def generate_text_output(self) -> TextOutput:
        """Generate a human-readable text output from the neural network.
        
        Returns:
            Text representation of the network's current state
        """
        pass
    
    def update_state(self, parameters: Dict[str, Any]) -> None:
        """Update the state of the neural network.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        self.state.parameters.update(parameters)
        self.state.last_update = datetime.now()
        
    def update_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """Update the developmental stage of the network.
        
        This method adjusts internal weights and parameters based on 
        the new developmental stage.
        
        Args:
            stage: New developmental stage
        """
        self.developmental_stage = stage
        
        # Update developmental weights
        new_weights = {
            DevelopmentalStage.INFANT: 0.2,
            DevelopmentalStage.TODDLER: 0.4,
            DevelopmentalStage.CHILD: 0.6, 
            DevelopmentalStage.ADOLESCENT: 0.8,
            DevelopmentalStage.MATURE: 1.0
        }
        
        # Set weight for current stage and above to the corresponding value
        for s in DevelopmentalStage:
            if s.value <= stage.value:
                self.state.developmental_weights[s] = new_weights[s]
            else:
                self.state.developmental_weights[s] = 0.1
                
        self.state.parameters["developmental_stage"] = stage.value
        
    def experiential_learning(self, input_data: torch.Tensor, target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
        """Learn from experience using the input data.
        
        This method implements a simple form of experiential learning, becoming more
        effective as the network develops.
        
        Args:
            input_data: Input tensor to learn from
            target: Optional target tensor for supervised learning
            
        Returns:
            Tuple of (output_tensor, loss_value)
        """
        # Forward pass
        output = self.forward(input_data)
        
        # If no target is provided, use a simple self-reinforcement approach
        if target is None:
            # Generate a pseudo-target by slightly enhancing strongest activations
            values, _ = torch.max(output, dim=1, keepdim=True)
            target = torch.where(output > 0.8 * values, output * 1.1, output * 0.9)
        
        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion.forward(output, target)
        
        # Scale learning by developmental stage
        effective_lr = self.learning_rate * self.state.developmental_weights[self.developmental_stage]
        
        # Backward pass and update weights - only if network is actively learning
        if self.training and effective_lr > 0:
            self.zero_grad()
            loss.backward()
            
            # Apply learning rate manually to make it development-dependent
            with torch.no_grad():
                for param in self.parameters():
                    if param.grad is not None:
                        param -= effective_lr * param.grad
        
        # Track this experience
        self.experience_count += 1
        self.last_activations.append(output.detach().mean().item())
        if len(self.last_activations) > 100:
            self.last_activations = self.last_activations[-100:]
            
        # Update state with training information
        self.update_state({
            "experience_count": self.experience_count,
            "last_loss": loss.item(),
            "average_activation": np.mean(self.last_activations)
        })
        
        return output, loss.item()
        
    def get_developmental_capacity(self) -> float:
        """Get the current developmental capacity of the network.
        
        Returns:
            Float representing developmental capacity (0-1)
        """
        return self.state.developmental_weights[self.developmental_stage]