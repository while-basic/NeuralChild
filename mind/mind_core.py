"""Core Mind class that manages the sub-neural networks."""

from typing import Dict, Any, List, Optional
from datetime import datetime

from neuralchild.core.schemas import NetworkMessage
from neuralchild.mind.schemas import MindState, ObservableState, Emotion, EmotionType
from neuralchild.core.neural_network import NeuralNetwork

class Mind:
    """Core class for the mind simulation."""
    
    def __init__(self):
        """Initialize the mind simulation."""
        self.networks: Dict[str, NeuralNetwork] = {}
        self.state = MindState(
            consciousness_level=0.5,
            emotional_state={},
            energy_level=0.7
        )
        self.messages: List[NetworkMessage] = []
        
    def register_network(self, network: NeuralNetwork) -> None:
        """Register a neural network with the mind."""
        self.networks[network.name] = network
        
    def process_input(self, input_data: Dict[str, Any]) -> None:
        """Process input data from the environment."""
        # TODO: Implement input processing logic
        pass
        
    def step(self) -> None:
        """Advance the mind simulation by one step."""
        # Process pending messages between networks
        self._process_messages()
        
        # Update network states
        for network in self.networks.values():
            network.generate_text_output()
            
        # Update overall mind state based on network states
        self._update_mind_state()
        
    def _process_messages(self) -> None:
        """Process messages between networks."""
        # Process each message in the queue
        for message in self.messages:
            if message.receiver in self.networks:
                self.networks[message.receiver].process_message(message)
        
        # Clear processed messages
        self.messages = []
        
    def _update_mind_state(self) -> None:
        """Update the overall mind state."""
        # Example: update based on network states
        # In a real implementation, this would aggregate data from all networks
        self.state.timestamp = datetime.now()
        
    def get_state(self) -> MindState:
        """Get the current state of the mind."""
        return self.state
        
    def get_observable_state(self) -> ObservableState:
        """Get the observable state of the mind."""
        # Calculate apparent mood from emotional state
        apparent_mood = sum([
            self.state.emotional_state.get(EmotionType.JOY, 0) * 1.0,
            self.state.emotional_state.get(EmotionType.TRUST, 0) * 0.8,
            self.state.emotional_state.get(EmotionType.SADNESS, 0) * -0.8,
            self.state.emotional_state.get(EmotionType.FEAR, 0) * -0.6,
            self.state.emotional_state.get(EmotionType.ANGER, 0) * -1.0
        ])
        
        # Clamp apparent mood to [-1, 1]
        apparent_mood = max(-1.0, min(1.0, apparent_mood))
        
        # Get recent emotions
        recent_emotions = [
            Emotion(name=name, intensity=intensity)
            for name, intensity in self.state.emotional_state.items()
            if intensity > 0.2  # Only include significant emotions
        ]
        
        # Calculate expressed needs based on current state
        expressed_needs = {
            "comfort": max(0, -apparent_mood),
            "play": self.state.energy_level if apparent_mood > 0 else 0,
            "rest": 1.0 - self.state.energy_level if self.state.energy_level < 0.3 else 0
        }
        
        return ObservableState(
            apparent_mood=apparent_mood,
            energy_level=self.state.energy_level,
            current_focus=self.state.current_focus,
            recent_emotions=recent_emotions,
            expressed_needs=expressed_needs
        )