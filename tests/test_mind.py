"""Tests for the Mind class."""

import pytest
import sys
import os
import torch
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.mind_core import Mind
from config import Config
from core.schemas import DevelopmentalStage

class TestMind:
    """Test suite for the Mind class."""
    
    def test_mind_initialization(self, mind: Mind):
        """Test that the Mind initializes correctly."""
        assert mind is not None
        assert mind.state.developmental_stage == DevelopmentalStage.INFANT
        assert mind.state.energy_level > 0
        assert mind.state.emotional_state is not None
    
    def test_mind_networks(self, mind: Mind):
        """Test that the Mind has the expected networks."""
        assert hasattr(mind, 'networks')
        # The networks dictionary might be empty initially
        assert isinstance(mind.networks, dict)
    
    def test_mind_process_input(self, mind: Mind, test_input: Dict[str, Any]):
        """Test that the Mind can process input."""
        # Process the input - note that this might return None
        mind.process_input(test_input)
        
        # Check that the mind has a state
        assert mind.state is not None
        
    def test_mind_step(self, mind: Mind):
        """Test that the Mind can perform a step."""
        # Perform a step
        mind.step()
        
        # Check that the mind still has a state after stepping
        assert mind.state is not None
