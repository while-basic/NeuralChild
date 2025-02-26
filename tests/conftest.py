"""Common test fixtures for the NeuralChild project."""

import pytest
import sys
import os
import torch
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from mind.mind_core import Mind
from mother.mother_llm import MotherLLM

@pytest.fixture
def config() -> Config:
    """Create a test configuration."""
    config = Config()
    # Set simulate_llm to True to avoid making real API calls during tests
    config.development.simulate_llm = True
    return config

@pytest.fixture
def mind() -> Mind:
    """Create a test Mind instance."""
    # The Mind class doesn't take a config parameter directly
    return Mind()

@pytest.fixture
def mother() -> MotherLLM:
    """Create a test MotherLLM instance."""
    # The MotherLLM class doesn't take a config parameter directly
    return MotherLLM()

@pytest.fixture
def test_input() -> Dict[str, Any]:
    """Create a test input for the Mind."""
    return {
        "visual": [0.1] * 64,
        "auditory": [0.2] * 64,
        "language": "Hello, child",
        "source": "test"
    }
