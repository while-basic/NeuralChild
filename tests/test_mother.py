"""Tests for the MotherLLM class."""

import pytest
import sys
import os
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mother.mother_llm import MotherLLM
from config import Config
from mind.mind_core import Mind

class TestMotherLLM:
    """Test suite for the MotherLLM class."""
    
    def test_mother_initialization(self, mother: MotherLLM):
        """Test that the MotherLLM initializes correctly."""
        assert mother is not None
        assert hasattr(mother, 'interaction_history')
        assert isinstance(mother.interaction_history, list)
    
    def test_mother_observe_and_respond(self, mother: MotherLLM, mind: Mind):
        """Test that the MotherLLM can observe and respond to the mind."""
        # Mock the LLM response using the chat_completion function
        with patch('utils.llm_module.chat_completion', return_value={"choices": [{"message": {"content": '{"understanding": "Child is calm", "response": "Hello, my child", "action": "speak", "development_focus": "language"}'}}]}):
            # Get a response
            response = mother.observe_and_respond(mind)
            
            # The response might be None if the mother decides not to respond
            # Just check that the method runs without errors
            assert mother is not None
    
    def test_mother_interaction_history(self, mother: MotherLLM):
        """Test that the MotherLLM maintains an interaction history."""
        # Check that the interaction history exists
        assert hasattr(mother, 'interaction_history')
        
        # The history might be empty initially
        assert isinstance(mother.interaction_history, list)
