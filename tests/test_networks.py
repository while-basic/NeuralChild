"""Tests for the neural network classes."""

import pytest
import sys
import os
import torch
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.networks.consciousness import ConsciousnessNetwork
from mind.networks.emotions import EmotionsNetwork
from mind.networks.perception import PerceptionNetwork
from mind.networks.thoughts import ThoughtsNetwork
from config import Config

class TestNetworks:
    """Test suite for the neural network classes."""
    
    def test_consciousness_network(self, config: Config):
        """Test that the ConsciousnessNetwork initializes and processes correctly."""
        # Create network with default parameters
        network = ConsciousnessNetwork(input_dim=64, hidden_dim=128, output_dim=64)
        
        # Check initialization
        assert network is not None
        assert hasattr(network, 'rnn')
        assert hasattr(network, 'self_model')
        
        # Create test input
        test_input = torch.rand(1, network.input_dim)
        
        # Process input
        output = network.forward(test_input)
        
        # Check output
        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape[1] == network.output_dim
    
    def test_emotions_network(self, config: Config):
        """Test that the EmotionsNetwork initializes and processes correctly."""
        # Create network with default parameters
        network = EmotionsNetwork(input_dim=64, hidden_dim=128, output_dim=64)
        
        # Check initialization
        assert network is not None
        assert hasattr(network, 'emotion_processor')
        
        # Create test input
        test_input = torch.rand(1, network.input_dim)
        
        # Process input
        output = network.forward(test_input)
        
        # Check output
        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape[1] == network.output_dim
    
    def test_perception_network(self, config: Config):
        """Test that the PerceptionNetwork initializes and processes correctly."""
        # Create network with default parameters
        network = PerceptionNetwork(input_dim=128, hidden_dim=256, output_dim=64)
        
        # Check initialization
        assert network is not None
        assert hasattr(network, 'visual_processor')
        assert hasattr(network, 'auditory_processor')
        
        # Create test input
        test_input = torch.rand(1, network.input_dim)
        
        # Process input
        output = network.forward(test_input)
        
        # Check output
        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape[1] == network.output_dim
    
    def test_thoughts_network(self, config: Config):
        """Test that the ThoughtsNetwork initializes and processes correctly."""
        # Create network with default parameters
        network = ThoughtsNetwork(input_dim=64, hidden_dim=128, output_dim=64)
        
        # Check initialization
        assert network is not None
        assert hasattr(network, 'thought_generator')
        
        # Create test input
        test_input = torch.rand(1, network.input_dim)
        
        # Process input
        output = network.forward(test_input)
        
        # Check output
        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape[1] == network.output_dim
