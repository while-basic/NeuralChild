"""Mother LLM component that interacts with the mind simulation."""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel

from neuralchild.mind.mind_core import Mind
from neuralchild.mind.schemas import ObservableState
from neuralchild.utils.llm_module import chat_completion

class MotherResponse(BaseModel):
    """Response from the mother LLM."""
    understanding: str  # Mother's interpretation of the mind's state
    response: str       # Nurturing response to the mind
    action: str         # Specific action to take (comfort, teach, play, etc.)
    timestamp: datetime = datetime.now()

class MotherLLM:
    """Mother LLM component that interacts with the mind simulation."""
    
    def __init__(self):
        """Initialize the Mother LLM."""
        self.interaction_history: List[Dict[str, Any]] = []
        self.base_prompt = """
        You are a nurturing and attentive mother figure. You can only perceive external
        behaviors and must respond based on what you observe, not internal states.
        Your responses should be caring, supportive, and appropriate for the situation.
        """

    def observe_and_respond(self, mind: Mind) -> Optional[MotherResponse]:
        """Observe the mind's external state and provide a nurturing response."""
        observable_state = mind.get_observable_state()
        
        # Construct situation description from observable state
        situation = self._construct_situation(observable_state)
        
        # Get response from LLM with structured output
        response = chat_completion(
            system_prompt=self.base_prompt,
            user_prompt=situation,
            structured_output=True
        )
        
        if response:
            # Create a MotherResponse from the LLM response
            mother_response = MotherResponse(
                understanding=response["understanding"],
                response=response["response"],
                action=response["action"]
            )
            
            # Add to interaction history
            self.interaction_history.append({
                'observation': observable_state.model_dump(),
                'response': mother_response.model_dump(),
                'timestamp': datetime.now().isoformat()
            })
            
            return mother_response
            
        return None

    def _construct_situation(self, state: ObservableState) -> str:
        """Convert observable state into a natural description for the LLM."""
        description = []

        # Describe apparent mood
        mood_desc = "seems content"
        if state.apparent_mood < -0.3:
            mood_desc = "appears distressed"
        elif state.apparent_mood > 0.3:
            mood_desc = "looks happy"
        description.append(f"The child {mood_desc}.")

        # Describe energy level
        if state.energy_level < 0.3:
            description.append("They seem tired and low in energy.")
        elif state.energy_level > 0.7:
            description.append("They are very energetic and active.")

        # Describe focus/attention
        if state.current_focus:
            description.append(f"They are focused on {state.current_focus}.")

        # Describe recent emotions
        if state.recent_emotions:
            emotion_desc = ", ".join(
                f"showing {e.name} (intensity: {e.intensity:.1f})"
                for e in state.recent_emotions
            )
            description.append(f"Recently, they have been {emotion_desc}.")

        # Describe expressed needs
        if state.expressed_needs:
            needs_desc = ", ".join(
                f"seeking {need}"
                for need in state.expressed_needs.keys()
            )
            description.append(f"The child appears to be {needs_desc}.")

        return " ".join(description)