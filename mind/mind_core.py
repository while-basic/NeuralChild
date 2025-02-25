"""Core Mind class that manages the sub-neural networks.

This module implements the central coordinator for all neural networks,
managing communication, development, and the overall state of the artificial mind.
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import random
import numpy as np
import torch
import logging

from core.schemas import (
    NetworkMessage, 
    Memory, 
    Belief, 
    Need, 
    DevelopmentalStage
)
from mind.schemas import (
    MindState, 
    ObservableState, 
    Emotion, 
    EmotionType,
    LanguageAbility
)
from core.neural_network import NeuralNetwork
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Mind:
    """Core class for the mind simulation.
    
    The Mind coordinates all neural networks, manages development,
    and maintains the overall state of the artificial mind.
    """
    
    def __init__(self):
        """Initialize the mind simulation."""
        self.networks: Dict[str, NeuralNetwork] = {}
        self.state = MindState(
            consciousness_level=0.2,  # Start with lower consciousness (infant-like)
            emotional_state={
                EmotionType.JOY: 0.3,
                EmotionType.TRUST: 0.3,
                EmotionType.FEAR: 0.3,
                EmotionType.SURPRISE: 0.3
            },
            energy_level=0.7,
            developmental_stage=DevelopmentalStage.INFANT,
            language_ability=LanguageAbility(
                vocabulary_size=0,
                sentence_complexity=0.0,
                understanding_level=0.1,
                expression_level=0.0
            )
        )
        
        # Messages queue for inter-network communication
        self.messages: List[NetworkMessage] = []
        
        # Initialize memory systems
        self.short_term_memory: List[Memory] = []
        self.long_term_memory: List[Memory] = []
        self.beliefs: List[Belief] = []
        
        # Initialize needs
        self.needs: Dict[str, Need] = {
            "comfort": Need(name="comfort", intensity=0.7),
            "stimulation": Need(name="stimulation", intensity=0.8),
            "rest": Need(name="rest", intensity=0.3),
            "bonding": Need(name="bonding", intensity=0.9),
            "autonomy": Need(name="autonomy", intensity=0.1)
        }
        
        # Development tracking
        self.developmental_milestones = {
            "emotions_experienced": set(),
            "vocabulary_learned": set(),
            "beliefs_formed": 0,
            "interactions_count": 0,
            "memories_formed": 0
        }
        
        # Development thresholds for advancing to next stage
        self.development_thresholds = {
            DevelopmentalStage.INFANT: {
                "emotions_experienced": 3,  # Number of distinct emotions experienced
                "interactions_count": 20,
                "memories_formed": 10
            },
            DevelopmentalStage.TODDLER: {
                "emotions_experienced": 5,
                "vocabulary_learned": 20,
                "interactions_count": 50,
                "memories_formed": 30
            },
            DevelopmentalStage.CHILD: {
                "emotions_experienced": 7,
                "vocabulary_learned": 100,
                "beliefs_formed": 10,
                "interactions_count": 100,
                "memories_formed": 100
            },
            DevelopmentalStage.ADOLESCENT: {
                "emotions_experienced": 8,
                "vocabulary_learned": 500,
                "beliefs_formed": 50,
                "interactions_count": 200,
                "memories_formed": 200
            }
        }
        
        self.last_developmental_check = datetime.now()
        self.last_need_update = datetime.now()
        self.last_memory_consolidation = datetime.now()
        self.simulation_time = 0.0  # Time in seconds since simulation start
        
        logger.info("Mind initialized at infant developmental stage")
        
    def register_network(self, network: NeuralNetwork) -> None:
        """Register a neural network with the mind.
        
        Args:
            network: Neural network to register
        """
        self.networks[network.name] = network
        network.update_developmental_stage(self.state.developmental_stage)
        logger.info(f"Registered network: {network.name}")
        
    def send_message(self, message: NetworkMessage) -> None:
        """Send a message between networks.
        
        Args:
            message: Message to send
        """
        # Set message developmental stage to current mind stage
        message.developmental_stage = self.state.developmental_stage
        self.messages.append(message)
        
    def process_input(self, input_data: Dict[str, Any]) -> None:
        """Process input data from the environment.
        
        Args:
            input_data: Dictionary of input data
        """
        # Route input to appropriate networks based on type
        if "visual" in input_data and "perception" in self.networks:
            visual_data = torch.tensor(input_data["visual"], dtype=torch.float32)
            self.networks["perception"].experiential_learning(visual_data)
            
        if "auditory" in input_data and "perception" in self.networks:
            auditory_data = torch.tensor(input_data["auditory"], dtype=torch.float32)
            self.networks["perception"].experiential_learning(auditory_data)
            
        if "language" in input_data and "language" in self.networks:
            # Process language input - helps with language acquisition
            language_data = input_data["language"]
            
            # Convert text to appropriate tensor representation
            if isinstance(language_data, str):
                # Very simple tokenization for demonstration
                tokens = language_data.lower().split()
                # Add words to vocabulary
                self.developmental_milestones["vocabulary_learned"].update(tokens)
                
                # As development progresses, language processing becomes more sophisticated
                if self.state.developmental_stage.value >= DevelopmentalStage.TODDLER.value:
                    # Simple numeric representation of tokens
                    tensor_data = torch.zeros(len(tokens), 10)  # Simple embedding
                    for i, token in enumerate(tokens):
                        # Hash the token to get a consistent embedding
                        hash_val = hash(token) % 1000
                        tensor_data[i] = torch.tensor([int(d) for d in f"{hash_val:010}"])
                    
                    if "language" in self.networks:
                        self.networks["language"].experiential_learning(tensor_data)
            
        # Form a memory of this input
        self._form_memory({
            "type": "sensory_input",
            "data": input_data,
            "time": datetime.now().isoformat()
        })
        
        # Increment interaction count
        self.developmental_milestones["interactions_count"] += 1
        
    def step(self) -> None:
        """Advance the mind simulation by one step."""
        start_time = datetime.now()
        
        # Process pending messages between networks
        self._process_messages()
        
        # Update network states
        for network in self.networks.values():
            # Generate text output for observability
            network.generate_text_output()
            
            # Each network gets a chance to do autonomous processing
            if hasattr(network, "autonomous_step"):
                network.autonomous_step()
        
        # Update needs
        self._update_needs()
        
        # Consolidate memories periodically
        self._consolidate_memories()
        
        # Update overall mind state based on network states
        self._update_mind_state()
        
        # Check for developmental progress
        self._check_developmental_progress()
        
        # Increment simulation time
        step_duration = (datetime.now() - start_time).total_seconds()
        self.simulation_time += step_duration
        
    def _process_messages(self) -> None:
        """Process messages between networks."""
        # Sort messages by priority
        self.messages.sort(key=lambda m: m.priority, reverse=True)
        
        # Process each message in the queue
        processed_messages = []
        for message in self.messages:
            # Only process messages appropriate for the current developmental stage
            if message.developmental_stage.value <= self.state.developmental_stage.value:
                if message.receiver in self.networks:
                    response = self.networks[message.receiver].process_message(message)
                    if response:
                        # If the network generated a response, add it to processed outcomes
                        logger.debug(f"Network {message.receiver} responded to message from {message.sender}")
                    processed_messages.append(message)
                elif message.receiver == "mind":
                    # Messages to the mind itself
                    self._process_mind_message(message)
                    processed_messages.append(message)
                else:
                    logger.warning(f"Message sent to unknown network: {message.receiver}")
        
        # Remove processed messages from queue
        for message in processed_messages:
            if message in self.messages:
                self.messages.remove(message)
        
    def _process_mind_message(self, message: NetworkMessage) -> None:
        """Process a message directed to the mind itself.
        
        Args:
            message: Message to process
        """
        if message.message_type == "emotion":
            # Update emotional state
            if "emotion" in message.content and "intensity" in message.content:
                emotion = message.content["emotion"]
                intensity = float(message.content["intensity"])
                
                try:
                    emotion_type = EmotionType(emotion)
                    self.state.emotional_state[emotion_type] = intensity
                    # Add to experienced emotions for developmental tracking
                    self.developmental_milestones["emotions_experienced"].add(emotion_type)
                    logger.debug(f"Updated emotion {emotion} to intensity {intensity}")
                except ValueError:
                    logger.warning(f"Unknown emotion type: {emotion}")
                    
        elif message.message_type == "belief":
            # Add or update a belief
            if all(k in message.content for k in ["subject", "predicate", "object", "confidence"]):
                subject = message.content["subject"]
                predicate = message.content["predicate"]
                obj = message.content["object"]
                confidence = float(message.content["confidence"])
                
                # Check if belief already exists
                for belief in self.beliefs:
                    if belief.subject == subject and belief.predicate == predicate and belief.object == obj:
                        # Update existing belief
                        belief.update_confidence(confidence)
                        return
                
                # Add new belief
                self.beliefs.append(Belief(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence=confidence,
                    developmental_stage=self.state.developmental_stage
                ))
                self.developmental_milestones["beliefs_formed"] += 1
                
        elif message.message_type == "consciousness":
            # Update consciousness level
            if "level" in message.content:
                self.state.consciousness_level = min(1.0, max(0.0, float(message.content["level"])))
                
        elif message.message_type == "need":
            # Update a need
            if "name" in message.content and "change" in message.content:
                need_name = message.content["name"]
                change = float(message.content["change"])
                
                if need_name in self.needs:
                    if "satisfy" in message.content and message.content["satisfy"]:
                        self.needs[need_name].satisfy(change)
                    else:
                        self.needs[need_name].update_intensity(change)
    
    def _update_needs(self) -> None:
        """Update the intensity of needs based on time and state."""
        current_time = datetime.now()
        elapsed = (current_time - self.last_need_update).total_seconds()
        
        if elapsed < config.mind.need_update_interval:
            return
        
        # Natural increase in needs over time
        self.needs["comfort"].update_intensity(0.01 * elapsed)
        self.needs["stimulation"].update_intensity(0.02 * elapsed)
        self.needs["rest"].update_intensity(0.005 * elapsed * (1.0 - self.state.energy_level))
        self.needs["bonding"].update_intensity(0.015 * elapsed)
        
        # Autonomy need grows with developmental stage
        autonomy_factor = max(0.001, (self.state.developmental_stage.value - 1) * 0.01)
        self.needs["autonomy"].update_intensity(autonomy_factor * elapsed)
        
        self.last_need_update = current_time
    
    def _form_memory(self, content: Dict[str, Any]) -> None:
        """Form a new short-term memory.
        
        Args:
            content: Memory content
        """
        # Add emotional context
        emotional_context = {
            emotion.name: intensity 
            for emotion, intensity in self.state.emotional_state.items()
            if intensity > 0.2  # Only include significant emotions
        }
        
        # Calculate emotional valence (-1 to 1)
        valence = sum([
            self.state.emotional_state.get(EmotionType.JOY, 0) * 1.0,
            self.state.emotional_state.get(EmotionType.TRUST, 0) * 0.8,
            self.state.emotional_state.get(EmotionType.SADNESS, 0) * -0.8,
            self.state.emotional_state.get(EmotionType.FEAR, 0) * -0.6,
            self.state.emotional_state.get(EmotionType.ANGER, 0) * -1.0
        ])
        valence = max(-1.0, min(1.0, valence))
        
        # Create memory with unique ID
        memory_id = f"mem_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
        
        memory = Memory(
            id=memory_id,
            content={
                **content,
                "emotional_context": emotional_context,
                "consciousness_level": self.state.consciousness_level
            },
            emotional_valence=valence,
            developmental_stage=self.state.developmental_stage,
            tags=["recent"]
        )
        
        # Add to short-term memory
        self.short_term_memory.append(memory)
        
        # Limit short-term memory size based on developmental stage
        max_stm_size = 3 + (self.state.developmental_stage.value * 2)
        if len(self.short_term_memory) > max_stm_size:
            self.short_term_memory = self.short_term_memory[-max_stm_size:]
            
        self.developmental_milestones["memories_formed"] += 1
    
    def _consolidate_memories(self) -> None:
        """Consolidate short-term memories into long-term memory."""
        current_time = datetime.now()
        
        # Only consolidate periodically
        if (current_time - self.last_memory_consolidation).total_seconds() < config.mind.memory_consolidation_interval:
            return
            
        # Consolidate memories with enough strength or emotional significance
        memories_to_consolidate = []
        for memory in self.short_term_memory:
            # Memories with strong emotional valence or accessed multiple times are consolidated
            if (abs(memory.emotional_valence) > 0.6 or 
                memory.strength > 1.5 or 
                (current_time - memory.creation_time).total_seconds() > 300):  # 5 minutes old
                memories_to_consolidate.append(memory)
                
                # Tag important memories
                if abs(memory.emotional_valence) > 0.7:
                    memory.tags.append("emotionally_significant")
                if memory.strength > 2.0:
                    memory.tags.append("well_reinforced")
        
        # Move memories to long-term storage
        for memory in memories_to_consolidate:
            if memory in self.short_term_memory:
                self.short_term_memory.remove(memory)
                self.long_term_memory.append(memory)
                
        # Forget weak long-term memories (more aggressive at infant/toddler stages)
        forget_threshold = 0.1
        if self.state.developmental_stage == DevelopmentalStage.INFANT:
            forget_threshold = 0.3
        elif self.state.developmental_stage == DevelopmentalStage.TODDLER:
            forget_threshold = 0.2
            
        # Apply decay to all long-term memories
        memories_to_forget = []
        for memory in self.long_term_memory:
            # More important memories decay slower
            decay_rate = 0.01
            if "emotionally_significant" in memory.tags:
                decay_rate *= 0.5
            if "well_reinforced" in memory.tags:
                decay_rate *= 0.7
                
            memory.decay(decay_rate)
            
            if memory.strength < forget_threshold:
                memories_to_forget.append(memory)
                
        # Remove forgotten memories
        for memory in memories_to_forget:
            if memory in self.long_term_memory:
                self.long_term_memory.remove(memory)
                
        self.last_memory_consolidation = current_time
        
    def _update_mind_state(self) -> None:
        """Update the overall mind state based on network and need states."""
        # Update energy level based on rest need
        rest_deficit = self.needs["rest"].intensity
        self.state.energy_level = max(0.1, min(1.0, 1.0 - (rest_deficit * 0.5)))
        
        # Update current focus based on most active network
        most_active_network = None
        highest_activation = 0.0
        
        for name, network in self.networks.items():
            if "average_activation" in network.state.parameters:
                activation = network.state.parameters["average_activation"]
                if activation > highest_activation:
                    highest_activation = activation
                    most_active_network = name
                    
        if most_active_network:
            self.state.current_focus = most_active_network
            
        # Update language ability based on developmental stage
        vocab_size = len(self.developmental_milestones["vocabulary_learned"])
        
        # Language ability scales with developmental stage and vocabulary
        sentence_complexity = min(1.0, (self.state.developmental_stage.value - 1) * 0.25 + (vocab_size / 1000))
        understanding = min(1.0, (self.state.developmental_stage.value - 1) * 0.2 + (vocab_size / 800))
        expression = min(1.0, (self.state.developmental_stage.value - 1) * 0.18 + (vocab_size / 1200))
        
        self.state.language_ability = LanguageAbility(
            vocabulary_size=vocab_size,
            sentence_complexity=sentence_complexity,
            understanding_level=understanding,
            expression_level=expression
        )
        
        # Update timestamp
        self.state.timestamp = datetime.now()
        
    def _check_developmental_progress(self) -> None:
        """Check if the mind has progressed to the next developmental stage."""
        current_time = datetime.now()
        
        # Only check periodically
        if (current_time - self.last_developmental_check).total_seconds() < config.mind.development_check_interval:
            return
            
        # Can't progress beyond mature
        if self.state.developmental_stage == DevelopmentalStage.MATURE:
            return
            
        # Get thresholds for current stage
        next_stage_value = self.state.developmental_stage.value + 1
        next_stage = DevelopmentalStage(next_stage_value)
        
        current_thresholds = self.development_thresholds[self.state.developmental_stage]
        
        # Check if all thresholds are met
        all_met = True
        for metric, threshold in current_thresholds.items():
            current_value = 0
            
            if metric == "emotions_experienced":
                current_value = len(self.developmental_milestones["emotions_experienced"])
            elif metric == "vocabulary_learned":
                current_value = len(self.developmental_milestones["vocabulary_learned"])
            elif metric in self.developmental_milestones:
                current_value = self.developmental_milestones[metric]
                
            if current_value < threshold:
                all_met = False
                break
                
        if all_met:
            # Progress to next stage!
            self.state.developmental_stage = next_stage
            logger.info(f"Mind has advanced to {next_stage.name} stage!")
            
            # Update all networks
            for network in self.networks.values():
                network.update_developmental_stage(next_stage)
                
        self.last_developmental_check = current_time
        
    def get_state(self) -> MindState:
        """Get the current state of the mind.
        
        Returns:
            Current mind state
        """
        return self.state
        
    def get_observable_state(self) -> ObservableState:
        """Get the observable state of the mind.
        
        Creates a representation of what would be externally observable
        about the mind's state, rather than its internal state.
        
        Returns:
            Observable state
        """
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
        
        # Calculate expressed needs based on current needs
        expressed_needs = {
            name: need.intensity
            for name, need in self.needs.items()
            if need.intensity > 0.5  # Only include significant needs
        }
        
        # Determine vocalization appropriate for developmental stage
        vocalization = self._generate_age_appropriate_vocalization()
        
        return ObservableState(
            apparent_mood=apparent_mood,
            energy_level=self.state.energy_level,
            current_focus=self.state.current_focus,
            recent_emotions=recent_emotions,
            expressed_needs=expressed_needs,
            developmental_stage=self.state.developmental_stage,
            vocalization=vocalization,
            age_appropriate_behaviors=self._get_age_appropriate_behaviors()
        )
        
    def _generate_age_appropriate_vocalization(self) -> str:
        """Generate an age-appropriate vocalization based on developmental stage.
        
        Returns:
            Age-appropriate vocalization
        """
        if self.state.developmental_stage == DevelopmentalStage.INFANT:
            # Infant vocalizations - cries, coos, etc.
            sounds = ["cries", "coos", "babbles", "gurgles", "fusses"]
            return random.choice(sounds)
            
        elif self.state.developmental_stage == DevelopmentalStage.TODDLER:
            # Toddler speech - single words and simple phrases
            vocab = list(self.developmental_milestones["vocabulary_learned"])
            if not vocab:
                return "babbles simple syllables"
                
            if len(vocab) < 5 or random.random() < 0.5:
                # Single word
                return f"says \"{random.choice(vocab)}\""
            else:
                # Simple phrase (2-3 words)
                phrase_len = min(3, len(vocab))
                phrase = " ".join(random.sample(vocab, phrase_len))
                return f"says \"{phrase}\""
                
        elif self.state.developmental_stage == DevelopmentalStage.CHILD:
            # Child speech - simple sentences
            vocab = list(self.developmental_milestones["vocabulary_learned"])
            if len(vocab) < 10:
                # Fall back to toddler speech
                return self._generate_age_appropriate_vocalization()
                
            # Simple sentence templates
            templates = [
                "I want {}",
                "I like {}",
                "I see {}",
                "Can I have {}?",
                "Where is {}?",
                "This is {}",
                "{} is mine",
                "I don't like {}"
            ]
            
            template = random.choice(templates)
            words = random.sample(vocab, min(3, len(vocab)))
            object_phrase = " ".join(words)
            return f"says \"{template.format(object_phrase)}\""
            
        elif self.state.developmental_stage == DevelopmentalStage.ADOLESCENT:
            # More complex sentences
            # (In a real implementation, this would use more sophisticated language generation)
            return "expresses thoughts in complete sentences"
            
        else:  # MATURE
            return "communicates fluently"
            
    def _get_age_appropriate_behaviors(self) -> List[str]:
        """Get a list of age-appropriate behaviors for the current developmental stage.
        
        Returns:
            List of behavior descriptions
        """
        behaviors = []
        
        if self.state.developmental_stage == DevelopmentalStage.INFANT:
            infant_behaviors = [
                "makes eye contact",
                "reaches for objects",
                "responds to voices",
                "shows interest in faces",
                "startles at loud noises",
                "smiles responsively",
                "tracks moving objects",
                "recognizes familiar faces"
            ]
            # Pick 1-2 random behaviors
            behaviors = random.sample(infant_behaviors, random.randint(1, 2))
            
        elif self.state.developmental_stage == DevelopmentalStage.TODDLER:
            toddler_behaviors = [
                "points at objects of interest",
                "imitates simple actions",
                "explores surroundings",
                "shows interest in peers",
                "expresses emotions more clearly",
                "attempts simple tasks",
                "shows preference for certain objects",
                "demonstrates simple problem-solving"
            ]
            # Pick 1-2 random behaviors
            behaviors = random.sample(toddler_behaviors, random.randint(1, 2))
            
        elif self.state.developmental_stage == DevelopmentalStage.CHILD:
            child_behaviors = [
                "engages in symbolic play",
                "follows simple instructions",
                "shows more complex emotions",
                "attempts to assert independence",
                "shows interest in rules and order",
                "asks many questions",
                "develops friendships",
                "demonstrates logical thinking"
            ]
            # Pick 1-2 random behaviors
            behaviors = random.sample(child_behaviors, random.randint(1, 2))
            
        elif self.state.developmental_stage == DevelopmentalStage.ADOLESCENT:
            adolescent_behaviors = [
                "shows abstract thinking",
                "contemplates hypothetical scenarios",
                "demonstrates complex emotional understanding",
                "shows more independence",
                "develops personal identity",
                "exhibits more complex social interactions",
                "shows interest in deeper concepts"
            ]
            # Pick 1-2 random behaviors
            behaviors = random.sample(adolescent_behaviors, random.randint(1, 2))
            
        else:  # MATURE
            mature_behaviors = [
                "demonstrates full emotional regulation",
                "exhibits complex reasoning",
                "shows nuanced social awareness",
                "displays integrated sense of self",
                "demonstrates abstract problem-solving",
                "exhibits creative thinking"
            ]
            # Pick 1-2 random behaviors
            behaviors = random.sample(mature_behaviors, random.randint(1, 2))
            
        return behaviors
    
    def save_state(self, directory: str = "saved_models") -> None:
        """Save the current state of the mind and all networks.
        
        Args:
            directory: Directory to save models
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save each network
            for name, network in self.networks.items():
                network_path = os.path.join(directory, f"{name}.pt")
                network.save_model(network_path)
            
            # Save mind state
            mind_state = {
                "developmental_stage": self.state.developmental_stage.value,
                "consciousness_level": self.state.consciousness_level,
                "emotional_state": {k.value: v for k, v in self.state.emotional_state.items()},
                "energy_level": self.state.energy_level,
                "simulation_time": self.simulation_time,
                "developmental_milestones": self.developmental_milestones,
                "needs": {k: {"intensity": v.intensity, "satisfaction_level": v.satisfaction_level} 
                        for k, v in self.needs.items()},
                "memory_count": len(self.long_term_memory),
                "belief_count": len(self.beliefs)
            }
            
            with open(os.path.join(directory, "mind_state.json"), "w") as f:
                json.dump(mind_state, f, indent=2)
                
            logger.info(f"Mind state saved to {directory}")
            
        except Exception as e:
            logger.error(f"Error saving mind state: {str(e)}")

    def load_state(self, directory: str = "saved_models") -> bool:
        """Load the mind state and all networks from disk.
        
        Args:
            directory: Directory to load models from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(directory):
            logger.warning(f"Save directory not found: {directory}")
            return False
            
        try:
            # Load mind state
            mind_state_path = os.path.join(directory, "mind_state.json")
            if os.path.exists(mind_state_path):
                with open(mind_state_path, "r") as f:
                    mind_state = json.load(f)
                    
                # Restore core state attributes
                if "developmental_stage" in mind_state:
                    self.state.developmental_stage = DevelopmentalStage(mind_state["developmental_stage"])
                    
                if "consciousness_level" in mind_state:
                    self.state.consciousness_level = mind_state["consciousness_level"]
                    
                if "energy_level" in mind_state:
                    self.state.energy_level = mind_state["energy_level"]
                    
                if "simulation_time" in mind_state:
                    self.simulation_time = mind_state["simulation_time"]
                    
                # Load networks
                for name, network in self.networks.items():
                    network_path = os.path.join(directory, f"{name}.pt")
                    if os.path.exists(network_path):
                        success = network.load_model(network_path)
                        if not success:
                            logger.warning(f"Failed to load network: {name}")
                    else:
                        logger.warning(f"Network model not found: {name}")
                        
                logger.info(f"Mind state loaded from {directory}")
                return True
                
        except Exception as e:
            logger.error(f"Error loading mind state: {str(e)}")
            
        return False