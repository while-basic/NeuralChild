"""Core schemas for the NeuralChild project."""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import torch

class NetworkMessage(BaseModel):
    """Message passed between neural networks."""
    sender: str
    receiver: str
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: float = 1.0
    
class NetworkState(BaseModel):
    """State of a neural network."""
    name: str
    active: bool = True
    last_update: datetime = Field(default_factory=datetime.now)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
class VectorOutput(BaseModel):
    """Vector output from a neural network."""
    source: str
    data: List[float]
    timestamp: datetime = Field(default_factory=datetime.now)
    
class TextOutput(BaseModel):
    """Text output from a neural network for human consumption."""
    source: str
    text: str
    confidence: float = 1.0
    timestamp: datetime = Field(default_factory=datetime.now)