"""Schemas for mind simulation components."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

class EmotionType(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    TRUST = "trust"
    ANTICIPATION = "anticipation"

class Emotion(BaseModel):
    """Representation of an emotion in the mind."""
    name: EmotionType
    intensity: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    
class MindState(BaseModel):
    """Overall state of the mind."""
    consciousness_level: float = Field(ge=0.0, le=1.0)
    emotional_state: Dict[EmotionType, float] = Field(default_factory=dict)
    current_focus: Optional[str] = None
    energy_level: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    
class ObservableState(BaseModel):
    """Observable state of the mind by external entities."""
    apparent_mood: float = Field(ge=-1.0, le=1.0)
    energy_level: float = Field(ge=0.0, le=1.0)
    current_focus: Optional[str] = None
    recent_emotions: List[Emotion] = Field(default_factory=list)
    expressed_needs: Dict[str, float] = Field(default_factory=dict)