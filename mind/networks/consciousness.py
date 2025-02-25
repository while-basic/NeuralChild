# neuralchild/mind/networks/consciousness.py
"""Consciousness neural network implementation."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from neuralchild.core.neural_network import NeuralNetwork
from neuralchild.core.schemas import NetworkMessage, VectorOutput, TextOutput

class ConsciousnessNetwork(NeuralNetwork):
    """
    Consciousness network that integrates awareness from other networks.
    
    Uses a recurrent neural network (RNN) architecture to maintain 
    a sense of continuity and awareness over time.
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 64):
        """Initialize the consciousness network."""
        super().__init__(name="consciousness", input_dim=input_dim, output_dim=output_dim)
        
        # RNN for processing sequential inputs
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Hidden state
        self.hidden = None
        
        # Current awareness level (0-1)
        self.awareness_level = 0.5
        
        # Initialize state parameters
        self.state.parameters.update({
            "awareness_level": self.awareness_level,
            "attending_to": None,
            "recent_inputs": []
        })
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network."""
        # Ensure input is 3D for RNN [batch, seq_len, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Initialize hidden state if None
        if self.hidden is None:
            self.hidden = torch.zeros(2, x.size(0), 128, device=x.device)
        
        # Process through RNN
        output, self.hidden = self.rnn(x, self.hidden)
        
        # Get last output
        last_output = output[:, -1, :]
        
        # Project to output dimension
        result = self.output_layer(last_output)
        
        # Update awareness level based on output activation
        self.awareness_level = torch.sigmoid(result.mean()).item()
        
        # Update state
        self.update_state({
            "awareness_level": self.awareness_level,
            "recent_inputs": self.state.parameters.get("recent_inputs", [])[-4:] + [x.detach().mean().item()]
        })
        
        return result
        
    def process_message(self, message: NetworkMessage) -> Optional[VectorOutput]:
        """Process a message from another neural network."""
        # Extract vector data from message content
        if "vector_data" in message.content and len(message.content["vector_data"]) == self.input_dim:
            # Convert to tensor and process
            input_tensor = torch.tensor(message.content["vector_data"], dtype=torch.float32)
            output_tensor = self.forward(input_tensor.unsqueeze(0))
            
            # Update state to reflect attention to the sender
            self.update_state({"attending_to": message.sender})
            
            # Return vector output
            return VectorOutput(
                source=self.name,
                data=output_tensor[0].tolist()
            )
        return None
        
    def generate_text_output(self) -> TextOutput:
        """Generate a human-readable text output from the neural network."""
        # Generate text based on current state
        awareness_text = "fully aware" if self.awareness_level > 0.8 else \
                        "aware" if self.awareness_level > 0.5 else \
                        "partially aware" if self.awareness_level > 0.2 else \
                        "barely aware"
        
        attending_to = self.state.parameters.get("attending_to", None)
        attending_text = f" and focusing on {attending_to}" if attending_to else ""
        
        # Create text output
        text = f"Consciousness is {awareness_text}{attending_text}."
        
        return TextOutput(
            source=self.name,
            text=text,
            confidence=self.awareness_level
        )