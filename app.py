"""
Neural Child - Streamlit Dashboard
==================================
Interactive visualization dashboard for the Neural Child project.
This application provides a comprehensive graphical interface for monitoring
and interacting with the artifical mind simulation.

Usage:
    streamlit run app.py [-- --config CONFIG_PATH]
"""

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import threading
import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import queue
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, List, Optional

# Neural Child imports
from config import load_config, Config, get_config
from mind.mind_core import Mind
from mother.mother_llm import MotherLLM
from mind.networks.consciousness import ConsciousnessNetwork
from mind.networks.emotions import EmotionsNetwork
from mind.networks.perception import PerceptionNetwork
from mind.networks.thoughts import ThoughtsNetwork
from core.schemas import DevelopmentalStage
from mind.schemas import EmotionType, Emotion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("neuralchild.streamlit")

# Global variables
CONFIG_PATH = None
running = False
simulation_thread = None
message_queue = queue.Queue()
simulation_metrics = {
    "iterations": 0,
    "start_time": None,
    "mother_interactions": 0,
    "developmental_milestones": {},
    "timestamped_data": []
}

# Configure exception handling for threads
def thread_exception_handler(args):
    """Handle exceptions in threads to avoid silent failures"""
    logger.error(f"Unhandled exception in thread: {args.exc_type}: {args.exc_value}")
    logger.error(f"Thread traceback: {args.exc_traceback}")
    
threading.excepthook = thread_exception_handler

# Colors for different developmental stages
STAGE_COLORS = {
    "INFANT": "#9bc2e6",     # Light blue
    "TODDLER": "#c4e0b2",    # Light green
    "CHILD": "#fed966",      # Yellow
    "ADOLESCENT": "#f9bab5", # Light red
    "MATURE": "#b3a2c7"      # Light purple
}

# Main emotion colors
EMOTION_COLORS = {
    "joy": "#ffce56",        # Yellow
    "sadness": "#36a2eb",    # Blue
    "anger": "#ff6384",      # Red
    "fear": "#9966ff",       # Purple
    "disgust": "#4bc0c0",    # Teal
    "surprise": "#ff9f40",   # Orange
    "trust": "#97bbcd",      # Light blue
    "anticipation": "#ffab91", # Salmon
    "confusion": "#ce93d8",  # Lavender
    "interest": "#80cbc4",   # Mint
    "boredom": "#bcaaa4"     # Taupe
}

def initialize_session_state():
    """Initialize Streamlit session state variables.
    
    This ensures all required session state variables are created before
    any other part of the app tries to access them.
    """
    # Use a dictionary to define all session state variables
    # This is more reliable than individual checks
    defaults = {
        'config': get_config(),
        'mind': None,
        'mother': None,
        'running': False,
        'observable_state': None,
        'mother_response': None,
        'interaction_history': [],
        'metrics_history': {
            'energy': [],
            'mood': [],
            'consciousness': [],
            'interactions': [],
            'emotions': [],
            'needs': [],
            'time_series': []
        },
        'developmental_metrics': {
            'emotions_experienced': 0,
            'vocabulary_learned': 0,
            'beliefs_formed': 0,
            'interactions_count': 0,
            'memories_formed': 0,
            'stage_time': datetime.now()
        },
        'network_outputs': {},
        'thread_created': False  # Add flag to track if thread was already created
    }
    
    # Initialize all session state variables
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Neural Child Streamlit Dashboard")
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    
    # Due to how Streamlit handles command line arguments, we need to check if -- is in sys.argv
    # to separate Streamlit arguments from our own
    if '--' in sys.argv:
        # Get arguments after --
        args_idx = sys.argv.index('--') + 1
        args = parser.parse_args(sys.argv[args_idx:])
    else:
        args = parser.parse_args([])
        
    return args

def load_project_configuration(config_path=None):
    """Load project configuration."""
    global CONFIG_PATH
    CONFIG_PATH = config_path
    
    # Load configuration
    config = load_config(config_path) if config_path else load_config()
    st.session_state.config = config
    
    return config

def initialize_simulation():
    """Initialize the Neural Child simulation."""
    try:
        # Initialize mind and mother
        mind = Mind()
        mother = MotherLLM()
        
        # Initialize neural networks
        initialize_networks(mind, st.session_state.config)
        
        # Store in session state
        st.session_state.mind = mind
        st.session_state.mother = mother
        
        # Reset thread created flag
        st.session_state.thread_created = False
        
        # Initialize metrics
        simulation_metrics["iterations"] = 0
        simulation_metrics["start_time"] = datetime.now()
        simulation_metrics["mother_interactions"] = 0
        simulation_metrics["developmental_milestones"] = {}
        simulation_metrics["timestamped_data"] = []
        
        # Clear history
        st.session_state.interaction_history = []
        st.session_state.metrics_history = {
            'energy': [],
            'mood': [],
            'consciousness': [],
            'interactions': [],
            'emotions': [],
            'needs': [],
            'time_series': []
        }
        
        st.session_state.developmental_metrics = {
            'emotions_experienced': 0,
            'vocabulary_learned': 0,
            'beliefs_formed': 0,
            'interactions_count': 0,
            'memories_formed': 0,
            'stage_time': datetime.now()
        }
        
        # Set initial observable state
        st.session_state.observable_state = mind.get_observable_state()
        
        # Initialize network outputs dictionary
        st.session_state.network_outputs = {}
        for name in mind.networks:
            st.session_state.network_outputs[name] = mind.networks[name].generate_text_output().text
        
        logger.info("Simulation initialized")
        return mind, mother
        
    except Exception as e:
        # Better error handling
        logger.error(f"Error initializing simulation: {str(e)}", exc_info=True)
        st.error(f"Failed to initialize simulation: {str(e)}")
        return None, None

def initialize_networks(mind, config):
    """Initialize and register neural networks with the mind."""
    network_configs = config.mind.networks
    
    # Initialize consciousness network
    consciousness_config = network_configs.get("consciousness", {})
    consciousness = ConsciousnessNetwork(
        input_dim=consciousness_config.get("input_dim", 64),
        hidden_dim=consciousness_config.get("hidden_dim", 128),
        output_dim=consciousness_config.get("output_dim", 64)
    )
    mind.register_network(consciousness)
    
    # Initialize emotions network
    emotions_config = network_configs.get("emotions", {})
    emotions = EmotionsNetwork(
        input_dim=emotions_config.get("input_dim", 32),
        hidden_dim=emotions_config.get("hidden_dim", 64),
        output_dim=emotions_config.get("output_dim", 32)
    )
    mind.register_network(emotions)
    
    # Initialize perception network
    perception_config = network_configs.get("perception", {})
    perception = PerceptionNetwork(
        input_dim=perception_config.get("input_dim", 128),
        hidden_dim=perception_config.get("hidden_dim", 256),
        output_dim=perception_config.get("output_dim", 64)
    )
    mind.register_network(perception)
    
    # Initialize thoughts network
    thoughts_config = network_configs.get("thoughts", {})
    thoughts = ThoughtsNetwork(
        input_dim=thoughts_config.get("input_dim", 64),
        hidden_dim=thoughts_config.get("hidden_dim", 128),
        output_dim=thoughts_config.get("output_dim", 64)
    )
    mind.register_network(thoughts)
    
    # Set starting developmental stage if specified
    if config.mind.starting_stage != "INFANT":
        try:
            starting_stage = DevelopmentalStage[config.mind.starting_stage]
            mind.state.developmental_stage = starting_stage
            
            # Update all networks
            for network in mind.networks.values():
                network.update_developmental_stage(starting_stage)
                
            logger.info(f"Set starting developmental stage to {starting_stage.name}")
        except KeyError:
            logger.warning(f"Invalid starting stage: {config.mind.starting_stage}, using INFANT")

def simulation_loop(mind, mother):
    """Main simulation loop running in a separate thread.
    
    Args:
        mind: Mind object passed directly to avoid session state issues
        mother: MotherLLM object passed directly to avoid session state issues
    """
    global running, message_queue
    
    iteration = 0
    
    while running:
        try:
            iteration += 1
            simulation_metrics["iterations"] = iteration
            
            # Advance mind simulation
            mind.step()
            
            # Get observable state and generate mother response
            observable_state = mind.get_observable_state()
            st.session_state.observable_state = observable_state
            
            # Get network outputs
            network_outputs = {}
            for name, network in mind.networks.items():
                network_outputs[name] = network.generate_text_output().text
            
            # Update message queue with the latest state
            message_queue.put({
                "type": "state_update",
                "observable_state": observable_state,
                "network_outputs": network_outputs,
                "iteration": iteration
            })
            
            # Check for mother response
            response = mother.observe_and_respond(mind)
            if response:
                simulation_metrics["mother_interactions"] += 1
                message_queue.put({
                    "type": "mother_response",
                    "response": response
                })
                
                # Add to interaction history
                st.session_state.interaction_history.append({
                    "timestamp": datetime.now(),
                    "observable_state": observable_state.to_dict(),
                    "mother_response": response.to_dict()
                })
            
            # Update metrics
            current_time = datetime.now()
            developmental_stage = observable_state.developmental_stage.name
            metrics_snapshot = {
                "timestamp": current_time,
                "iteration": iteration,
                "stage": developmental_stage,
                "energy": observable_state.energy_level,
                "mood": observable_state.apparent_mood,
                "consciousness": mind.state.consciousness_level,
                "needs": {k: v for k, v in observable_state.expressed_needs.items()},
                "emotions": {e.name.value: e.intensity for e in observable_state.recent_emotions},
                "vocalization": observable_state.vocalization
            }
            simulation_metrics["timestamped_data"].append(metrics_snapshot)
            
            # Update developmental metrics
            update_developmental_metrics(mind)
            
            # Sleep for the configured step interval
            time.sleep(st.session_state.config.mind.step_interval)
            
        except Exception as e:
            logger.error(f"Error in simulation loop: {str(e)}", exc_info=True)
            message_queue.put({
                "type": "error",
                "error": str(e)
            })
            break

def update_developmental_metrics(mind):
    """Update developmental metrics from the mind."""
    developmental_milestones = mind.developmental_milestones
    
    st.session_state.developmental_metrics.update({
        'emotions_experienced': len(developmental_milestones.get('emotions_experienced', set())),
        'vocabulary_learned': len(developmental_milestones.get('vocabulary_learned', set())),
        'beliefs_formed': developmental_milestones.get('beliefs_formed', 0),
        'interactions_count': developmental_milestones.get('interactions_count', 0),
        'memories_formed': developmental_milestones.get('memories_formed', 0)
    })
    
    # Check if developmental stage has changed
    current_stage = mind.state.developmental_stage
    previous_stage = st.session_state.observable_state.developmental_stage if st.session_state.observable_state else current_stage
    
    if current_stage != previous_stage:
        st.session_state.developmental_metrics['stage_time'] = datetime.now()
        logger.info(f"Developmental stage changed to {current_stage.name}")

def start_simulation():
    """Start the simulation in a separate thread."""
    global running, simulation_thread
    
    if not st.session_state.mind or not st.session_state.mother:
        initialize_simulation()
    
    # Get references to mind and mother before threading
    mind = st.session_state.mind
    mother = st.session_state.mother
    
    running = True
    st.session_state.running = True
    
    # Start the simulation thread - pass mind and mother directly to avoid session state issues
    simulation_thread = threading.Thread(
        target=simulation_loop,
        args=(mind, mother)
    )
    simulation_thread.daemon = True
    simulation_thread.start()
    
    logger.info("Simulation started")

def stop_simulation():
    """Stop the simulation thread."""
    global running
    running = False
    st.session_state.running = False
    logger.info("Simulation stopped")

def restart_simulation():
    """Restart the simulation with the current configuration."""
    stop_simulation()
    time.sleep(0.5)  # Allow thread to stop
    initialize_simulation()
    start_simulation()
    logger.info("Simulation restarted")

def process_message_queue():
    """Process messages from the simulation thread."""
    global message_queue
    
    # Process all available messages
    while not message_queue.empty():
        try:
            message = message_queue.get_nowait()
            message_type = message.get("type")
            
            if message_type == "state_update":
                st.session_state.observable_state = message.get("observable_state")
                st.session_state.network_outputs = message.get("network_outputs")
                update_metrics_history(message.get("observable_state"))
                
            elif message_type == "mother_response":
                st.session_state.mother_response = message.get("response")
                
            elif message_type == "error":
                st.error(f"Simulation error: {message.get('error')}")
                stop_simulation()
                
        except queue.Empty:
            break
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)

def update_metrics_history(observable_state):
    """Update the metrics history with the latest state."""
    if not observable_state:
        return
        
    # Get the mind state for consciousness level
    mind = st.session_state.mind
    consciousness_level = mind.state.consciousness_level if mind else 0.0
    
    # Add current timestamp
    current_time = datetime.now()
    st.session_state.metrics_history['time_series'].append(current_time)
    
    # Add energy level
    st.session_state.metrics_history['energy'].append(observable_state.energy_level)
    
    # Add mood
    st.session_state.metrics_history['mood'].append(observable_state.apparent_mood)
    
    # Add consciousness level
    st.session_state.metrics_history['consciousness'].append(consciousness_level)
    
    # Add interaction count
    interactions = simulation_metrics.get("mother_interactions", 0)
    st.session_state.metrics_history['interactions'].append(interactions)
    
    # Process emotions - create a dictionary for each timestamp
    emotions_dict = {}
    for emotion in observable_state.recent_emotions:
        emotions_dict[emotion.name.value] = emotion.intensity
    
    # Store the emotions dictionary for this timestamp
    st.session_state.metrics_history['emotions'].append(emotions_dict)
    
    # Process needs
    st.session_state.metrics_history['needs'].append(observable_state.expressed_needs)

def create_brain_network_visualization():
    """Create a network visualization of the brain's neural networks."""
    G = nx.Graph()
    
    # Add nodes for each network
    networks = ["consciousness", "emotions", "perception", "thoughts"]
    for network in networks:
        G.add_node(network)
    
    # Add edges between networks that communicate
    G.add_edge("perception", "consciousness")
    G.add_edge("perception", "emotions")
    G.add_edge("perception", "thoughts")
    G.add_edge("emotions", "consciousness")
    G.add_edge("emotions", "thoughts")
    G.add_edge("thoughts", "consciousness")
    
    # Create positions for nodes
    pos = {
        "consciousness": (0.5, 0.8),
        "emotions": (0.2, 0.4),
        "perception": (0.8, 0.4),
        "thoughts": (0.5, 0.2)
    }
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Define node colors based on network activity
    network_activity = {}
    mind = st.session_state.mind
    
    if mind and mind.networks:
        for name, network in mind.networks.items():
            activation = network.state.parameters.get("average_activation", 0.5)
            network_activity[name] = activation
    else:
        for network in networks:
            network_activity[network] = 0.5
    
    node_colors = [plt.cm.Blues(network_activity.get(n, 0.5)) for n in G.nodes()]
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    # Add current focus highlight if applicable
    observable_state = st.session_state.observable_state
    if observable_state and observable_state.current_focus:
        focus = observable_state.current_focus
        if focus in pos:
            nx.draw_networkx_nodes(G, pos, nodelist=[focus], node_size=3500, 
                                 node_color='yellow', alpha=0.3, ax=ax)
    
    ax.set_title("Neural Network Connections", fontsize=16)
    ax.axis('off')
    
    return fig

def display_dashboard():
    """Display the main dashboard."""
    st.title("Neural Child Brain Simulation")
    
    # Create tabs for different sections
    tabs = st.tabs([
        "📊 Dashboard", 
        "🧠 Neural Networks", 
        "👩‍👦 Mother-Child Interaction", 
        "📈 Development Metrics",
        "⚙️ Configuration"
    ])
    
    with tabs[0]:
        display_main_dashboard()
        
    with tabs[1]:
        display_neural_networks()
        
    with tabs[2]:
        display_interaction_tab()
        
    with tabs[3]:
        display_development_metrics()
        
    with tabs[4]:
        display_configuration()

def display_main_dashboard():
    """Display the main dashboard tab."""
    st.header("Neural Child Dashboard")
    
    # Top row with controls and status
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Simulation controls
        if not st.session_state.running:
            if st.button("▶️ Start Simulation", key="start_btn", use_container_width=True):
                try:
                    start_simulation()
                except Exception as e:
                    st.error(f"Failed to start simulation: {str(e)}")
                    logger.error(f"Error starting simulation: {str(e)}", exc_info=True)
        else:
            if st.button("⏹️ Stop Simulation", key="stop_btn", use_container_width=True):
                try:
                    stop_simulation()
                except Exception as e:
                    st.error(f"Failed to stop simulation: {str(e)}")
                    logger.error(f"Error stopping simulation: {str(e)}", exc_info=True)
                
        if st.button("🔄 Restart Simulation", key="restart_btn", use_container_width=True):
            try:
                restart_simulation()
            except Exception as e:
                st.error(f"Failed to restart simulation: {str(e)}")
                logger.error(f"Error restarting simulation: {str(e)}", exc_info=True)
    
    with col2:
        # Current stage and status
        observable_state = st.session_state.observable_state
        stage = observable_state.developmental_stage.name if observable_state else "Not Started"
        
        st.metric(
            label="Developmental Stage", 
            value=stage,
            delta="Running" if st.session_state.running else "Stopped"
        )
        
        # Display developmental description
        if observable_state:
            st.write(observable_state.get_developmental_description())
    
    with col3:
        # Simulation metrics
        iterations = simulation_metrics.get("iterations", 0)
        st.metric(label="Iterations", value=iterations)
        
        if simulation_metrics.get("start_time"):
            elapsed = datetime.now() - simulation_metrics.get("start_time")
            elapsed_str = str(timedelta(seconds=int(elapsed.total_seconds())))
            st.metric(label="Elapsed Time", value=elapsed_str)
            
    # Main state display
    st.subheader("Mind State")
    col1, col2, col3 = st.columns(3)
    
    observable_state = st.session_state.observable_state
    if observable_state:
        with col1:
            # Energy and mood
            st.progress(observable_state.energy_level, text=f"Energy Level: {observable_state.energy_level:.2f}")
            
            # Map mood from [-1, 1] to [0, 1] for progress bar
            mood_mapped = (observable_state.apparent_mood + 1) / 2
            mood_color = "green" if observable_state.apparent_mood > 0 else "orange" if observable_state.apparent_mood > -0.5 else "red"
            st.progress(mood_mapped, text=f"Mood: {observable_state.apparent_mood:.2f}")
        
        with col2:
            # Needs
            st.subheader("Current Needs")
            if observable_state.expressed_needs:
                for need, intensity in observable_state.expressed_needs.items():
                    st.progress(intensity, text=f"{need.capitalize()}: {intensity:.2f}")
            else:
                st.write("No significant needs at the moment.")
        
        with col3:
            # Behaviors and vocalization
            st.subheader("Behaviors")
            if observable_state.vocalization:
                st.info(f"**Vocalization:** {observable_state.vocalization}")
                
            if observable_state.age_appropriate_behaviors:
                for behavior in observable_state.age_appropriate_behaviors:
                    st.write(f"• {behavior}")
            else:
                st.write("No observable behaviors at the moment.")
    
    # Brain visualization and emotions
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Brain Network")
        if st.session_state.mind:
            fig = create_brain_network_visualization()
            st.pyplot(fig)
        else:
            st.write("Simulation not started. Press 'Start Simulation' to begin.")
    
    with col2:
        st.subheader("Emotional State")
        if observable_state and observable_state.recent_emotions:
            # Create data for emotions chart
            emotions_data = []
            for emotion in observable_state.recent_emotions:
                emotions_data.append({
                    "Emotion": emotion.name.value,
                    "Intensity": emotion.intensity,
                    "Color": EMOTION_COLORS.get(emotion.name.value, "#CCCCCC")
                })
                
            # Create dataframe
            emotions_df = pd.DataFrame(emotions_data)
            
            # Create horizontal bar chart with Altair
            if not emotions_df.empty:
                chart = alt.Chart(emotions_df).mark_bar().encode(
                    x='Intensity:Q',
                    y=alt.Y('Emotion:N', sort='-x'),
                    color=alt.Color('Color:N', scale=None),
                    tooltip=['Emotion', 'Intensity']
                ).properties(height=30 * len(emotions_df))
                
                st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No emotional data available yet.")
    
    # Bottom section with time series charts
    st.subheader("Real-time Metrics")
    
    if len(st.session_state.metrics_history['time_series']) > 1:
        # Create time series dataframe
        time_data = []
        for i, ts in enumerate(st.session_state.metrics_history['time_series']):
            if i < len(st.session_state.metrics_history['energy']) and i < len(st.session_state.metrics_history['mood']):
                time_data.append({
                    "Time": ts,
                    "Energy": st.session_state.metrics_history['energy'][i],
                    "Mood": st.session_state.metrics_history['mood'][i],
                    "Consciousness": st.session_state.metrics_history['consciousness'][i]
                })
                
        df = pd.DataFrame(time_data)
        
        # Melt the dataframe for plotting multiple lines
        df_melted = pd.melt(df, id_vars=['Time'], value_vars=['Energy', 'Mood', 'Consciousness'],
                         var_name='Metric', value_name='Value')
        
        # Create line chart with Altair
        chart = alt.Chart(df_melted).mark_line(point=True).encode(
            x='Time:T',
            y=alt.Y('Value:Q', scale=alt.Scale(domain=[-1, 1])),
            color='Metric:N',
            tooltip=['Time', 'Metric', 'Value']
        ).properties(height=300)
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Waiting for more data points to display time series charts...")

def display_neural_networks():
    """Display the neural networks tab."""
    st.header("Neural Networks")
    
    # Create expanders for each network
    networks = {
        "Consciousness Network": "consciousness",
        "Emotions Network": "emotions",
        "Perception Network": "perception",
        "Thoughts Network": "thoughts"
    }
    
    for display_name, network_name in networks.items():
        with st.expander(display_name, expanded=(network_name == "consciousness")):
            # Get network output
            output_text = st.session_state.network_outputs.get(network_name, "Network not initialized")
            st.info(output_text)
            
            # Get network parameters if available
            mind = st.session_state.mind
            if mind and mind.networks and network_name in mind.networks:
                network = mind.networks[network_name]
                params = network.state.parameters
                
                # Create two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Network Parameters")
                    # Display relevant parameters based on network type
                    if network_name == "consciousness":
                        awareness = params.get("awareness_level", 0)
                        st.progress(awareness, text=f"Awareness Level: {awareness:.2f}")
                        
                        self_awareness = params.get("self_awareness", 0)
                        st.progress(self_awareness, text=f"Self-Awareness: {self_awareness:.2f}")
                        
                        integration = params.get("integration_capacity", 0)
                        st.progress(integration, text=f"Integration Capacity: {integration:.2f}")
                        
                    elif network_name == "emotions":
                        reactivity = params.get("reactivity", 0)
                        st.progress(reactivity, text=f"Emotional Reactivity: {reactivity:.2f}")
                        
                        regulation = params.get("regulation", 0)
                        st.progress(regulation, text=f"Emotional Regulation: {regulation:.2f}")
                        
                        # Display emotional state
                        if "emotional_state" in params:
                            emotion_data = []
                            emotional_state = params["emotional_state"]
                            for emotion, intensity in emotional_state.items():
                                emotion_data.append({
                                    "Emotion": emotion,
                                    "Intensity": intensity,
                                    "Color": EMOTION_COLORS.get(emotion, "#CCCCCC")
                                })
                                
                            if emotion_data:
                                df = pd.DataFrame(emotion_data)
                                chart = alt.Chart(df).mark_bar().encode(
                                    x='Intensity:Q',
                                    y=alt.Y('Emotion:N', sort='-x'),
                                    color=alt.Color('Color:N', scale=None)
                                ).properties(height=25 * len(df))
                                
                                st.altair_chart(chart, use_container_width=True)
                        
                    elif network_name == "perception":
                        object_rec = params.get("object_recognition", 0)
                        st.progress(object_rec, text=f"Object Recognition: {object_rec:.2f}")
                        
                        pattern_rec = params.get("pattern_recognition", 0)
                        st.progress(pattern_rec, text=f"Pattern Recognition: {pattern_rec:.2f}")
                        
                        focus = params.get("attentional_focus", "none")
                        st.write(f"**Attentional Focus:** {focus}")
                        
                    elif network_name == "thoughts":
                        abstract = params.get("abstract_thinking", 0)
                        st.progress(abstract, text=f"Abstract Thinking: {abstract:.2f}")
                        
                        logical = params.get("logical_reasoning", 0)
                        st.progress(logical, text=f"Logical Reasoning: {logical:.2f}")
                        
                        creativity = params.get("creativity", 0)
                        st.progress(creativity, text=f"Creativity: {creativity:.2f}")
                        
                        # Display current thoughts
                        if "current_thoughts" in params:
                            st.subheader("Current Thoughts")
                            thoughts = params["current_thoughts"]
                            for thought in thoughts:
                                st.write(f"• {thought}")
                
                with col2:
                    st.subheader("Developmental Weights")
                    dev_weights = network.state.developmental_weights
                    dev_data = []
                    
                    for stage, weight in dev_weights.items():
                        dev_data.append({
                            "Stage": stage.name,
                            "Weight": weight,
                            "Color": STAGE_COLORS.get(stage.name, "#CCCCCC")
                        })
                        
                    df = pd.DataFrame(dev_data)
                    chart = alt.Chart(df).mark_bar().encode(
                        x='Weight:Q',
                        y=alt.Y('Stage:N', sort=['INFANT', 'TODDLER', 'CHILD', 'ADOLESCENT', 'MATURE']),
                        color=alt.Color('Color:N', scale=None)
                    ).properties(height=150)
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Network statistics
                    st.subheader("Network Statistics")
                    experience_count = params.get("experience_count", 0)
                    st.metric("Learning Experiences", experience_count)
                    
                    avg_activation = params.get("average_activation", 0)
                    st.metric("Average Activation", f"{avg_activation:.4f}")
                    
                    last_loss = params.get("last_loss", 0)
                    st.metric("Last Training Loss", f"{last_loss:.4f}" if last_loss else "N/A")
            else:
                st.write("Network not initialized or not available.")

def display_interaction_tab():
    """Display the mother-child interaction tab."""
    st.header("Mother-Child Interaction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Recent Interactions")
        
        # Display interaction history
        if not st.session_state.interaction_history:
            st.info("No interactions recorded yet. The simulation needs to run for a while before the mother responds.")
        else:
            # Display the most recent interactions (limited to last 10)
            interaction_history = st.session_state.interaction_history[-10:]
            for i, interaction in enumerate(reversed(interaction_history)):
                with st.container():
                    # Child's observable state
                    observable = interaction["observable_state"]
                    stage = observable["developmental_stage"]
                    stage_color = STAGE_COLORS.get(stage, "#CCCCCC")
                    
                    # Extract child's behavior
                    child_behavior = ""
                    if "vocalization" in observable and observable["vocalization"]:
                        child_behavior += f"Child {observable['vocalization']}. "
                        
                    if "age_appropriate_behaviors" in observable and observable["age_appropriate_behaviors"]:
                        behaviors = observable["age_appropriate_behaviors"]
                        child_behavior += f"Behaviors: {', '.join(behaviors)}."
                    
                    st.markdown(
                        f'<div style="background-color:{stage_color}30; padding:10px; border-radius:5px; margin-bottom:5px;">'
                        f'<p><strong>Child ({stage}):</strong> {child_behavior}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Mother's response
                    if "mother_response" in interaction:
                        mother_resp = interaction["mother_response"]
                        response_text = mother_resp.get("response", "")
                        action = mother_resp.get("action", "")
                        focus = mother_resp.get("development_focus", "general")
                        
                        st.markdown(
                            f'<div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom:15px;">'
                            f'<p><strong>Mother ({action}/{focus}):</strong> {response_text}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
    with col2:
        st.subheader("Mother's Current Response")
        
        # Display current mother response
        mother_response = st.session_state.mother_response
        if mother_response:
            st.write("**Understanding:**")
            st.info(mother_response.understanding)
            
            st.write("**Response:**")
            st.success(mother_response.response)
            
            st.write("**Action Type:**")
            st.write(mother_response.action.capitalize())
            
            if mother_response.development_focus:
                st.write("**Developmental Focus:**")
                st.write(mother_response.development_focus.capitalize())
                
            st.write("**Timestamp:**")
            st.write(mother_response.timestamp.strftime("%H:%M:%S"))
        else:
            st.info("Mother hasn't responded yet. The simulation needs to run for a while before the mother responds.")
            
        # Mother interaction metrics
        st.subheader("Interaction Metrics")
        
        interaction_count = simulation_metrics.get("mother_interactions", 0)
        st.metric("Total Mother Interactions", interaction_count)
        
        # Calculate interaction rate if simulation has been running
        if simulation_metrics.get("start_time"):
            elapsed = (datetime.now() - simulation_metrics.get("start_time")).total_seconds()
            if elapsed > 0:
                rate = interaction_count / (elapsed / 60)  # per minute
                st.metric("Interaction Rate", f"{rate:.1f} per minute")

def display_development_metrics():
    """Display the development metrics tab."""
    st.header("Development Metrics")
    
    # Top metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        observable_state = st.session_state.observable_state
        stage = observable_state.developmental_stage.name if observable_state else "Not Started"
        stage_color = STAGE_COLORS.get(stage, "#CCCCCC")
        
        # Style the display
        st.markdown(
            f'<div style="background-color:{stage_color}; padding:20px; border-radius:10px; text-align:center;">'
            f'<h2 style="margin:0; color:white;">{stage}</h2>'
            f'<p style="margin:0; color:white;">Current Developmental Stage</p>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        if observable_state:
            st.write(observable_state.get_developmental_description())
            
            # Calculate time in current stage
            stage_time = st.session_state.developmental_metrics.get('stage_time', datetime.now())
            time_in_stage = datetime.now() - stage_time
            st.metric("Time in Current Stage", str(timedelta(seconds=int(time_in_stage.total_seconds()))))
    
    with col2:
        # Developmental milestones progress
        st.subheader("Developmental Milestones")
        
        metrics = st.session_state.developmental_metrics
        
        # Get thresholds for current stage
        thresholds = {}
        if st.session_state.mind:
            current_stage = st.session_state.mind.state.developmental_stage
            thresholds = st.session_state.mind.development_thresholds.get(current_stage, {})
        
        # Display progress bars for each metric
        metrics_display = [
            ("Emotions Experienced", metrics.get('emotions_experienced', 0), thresholds.get('emotions_experienced', 10)),
            ("Vocabulary Learned", metrics.get('vocabulary_learned', 0), thresholds.get('vocabulary_learned', 100)),
            ("Beliefs Formed", metrics.get('beliefs_formed', 0), thresholds.get('beliefs_formed', 20)),
            ("Interactions Count", metrics.get('interactions_count', 0), thresholds.get('interactions_count', 50)),
            ("Memories Formed", metrics.get('memories_formed', 0), thresholds.get('memories_formed', 50))
        ]
        
        for label, value, threshold in metrics_display:
            if threshold > 0:
                progress = min(1.0, value / threshold)
                st.progress(progress, text=f"{label}: {value}/{threshold}")
            else:
                st.write(f"{label}: {value}")
    
    with col3:
        # Next stage requirements
        st.subheader("Next Stage Requirements")
        
        if st.session_state.mind:
            mind = st.session_state.mind
            current_stage = mind.state.developmental_stage
            
            # If not at MATURE stage yet
            if current_stage.value < DevelopmentalStage.MATURE.value:
                next_stage = DevelopmentalStage(current_stage.value + 1)
                st.write(f"Requirements to advance to **{next_stage.name}**:")
                
                # Get current thresholds
                thresholds = mind.development_thresholds.get(current_stage, {})
                
                for metric, threshold in thresholds.items():
                    current_value = 0
                    
                    if metric == "emotions_experienced":
                        current_value = len(mind.developmental_milestones.get("emotions_experienced", set()))
                    elif metric == "vocabulary_learned":
                        current_value = len(mind.developmental_milestones.get("vocabulary_learned", set()))
                    elif metric in mind.developmental_milestones:
                        current_value = mind.developmental_milestones.get(metric, 0)
                        
                    st.write(f"• {metric.replace('_', ' ').title()}: {current_value}/{threshold}")
            else:
                st.write("Already at MATURE stage (highest level).")
    
    # Development progress visualization
    st.subheader("Developmental Progress")
    
    if len(st.session_state.metrics_history['time_series']) > 0:
        # Create time-series dataframe for emotions
        emotion_series = []
        
        for i, ts in enumerate(st.session_state.metrics_history['time_series']):
            if i < len(st.session_state.metrics_history['emotions']):
                emotion_dict = st.session_state.metrics_history['emotions'][i]
                record = {"Time": ts}
                
                # Add each emotion to the record
                for emotion, intensity in emotion_dict.items():
                    record[emotion] = intensity
                
                emotion_series.append(record)
        
        if emotion_series:
            # Create a heatmap of emotions over time
            emotion_df = pd.DataFrame(emotion_series)
            
            # Melt the dataframe for heatmap
            if len(emotion_df.columns) > 1:  # Make sure we have emotions
                # Fill NA values with 0
                emotion_df = emotion_df.fillna(0)
                
                # Select and melt only emotion columns
                emotion_cols = [col for col in emotion_df.columns if col != "Time"]
                if emotion_cols:
                    emotion_melt = pd.melt(
                        emotion_df, 
                        id_vars=['Time'], 
                        value_vars=emotion_cols,
                        var_name='Emotion', 
                        value_name='Intensity'
                    )
                    
                    # Create heatmap with plotly
                    fig = px.density_heatmap(
                        emotion_melt, 
                        x='Time', 
                        y='Emotion', 
                        z='Intensity',
                        color_continuous_scale='Viridis',
                        title='Emotional Development Over Time'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Show developmental stage progression
    if st.session_state.interaction_history:
        st.subheader("Developmental Stage Progression")
        
        # Extract developmental stages over time
        stage_progression = []
        
        for interaction in st.session_state.interaction_history:
            ts = interaction["timestamp"]
            stage = interaction["observable_state"]["developmental_stage"]
            stage_progression.append({"Time": ts, "Stage": stage})
        
        stage_df = pd.DataFrame(stage_progression)
        
        # Create a scatter plot showing stage progression
        if not stage_df.empty:
            # Map stages to numeric values
            stage_map = {
                "INFANT": 1,
                "TODDLER": 2,
                "CHILD": 3,
                "ADOLESCENT": 4,
                "MATURE": 5
            }
            
            stage_df["StageValue"] = stage_df["Stage"].map(stage_map)
            
            # Create a line chart with plotly
            fig = px.line(
                stage_df, 
                x='Time', 
                y='StageValue',
                labels={"StageValue": "Developmental Stage", "Time": "Simulation Time"},
                title="Developmental Stage Progression"
            )
            
            # Update y-axis ticks
            fig.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=["INFANT", "TODDLER", "CHILD", "ADOLESCENT", "MATURE"]
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No developmental data available yet. The simulation needs to run longer.")

def display_configuration():
    """Display the configuration tab."""
    st.header("Simulation Configuration")
    
    # Configuration warning
    st.warning("⚠️ Changes to configuration will take effect when you restart the simulation.")
    
    # Create tabs for different configuration sections
    config_tabs = st.tabs([
        "General Settings", 
        "Mind Settings", 
        "Network Settings", 
        "Advanced Settings"
    ])
    
    config = st.session_state.config
    
    with config_tabs[0]:
        st.subheader("General Simulation Settings")
        
        # Starting developmental stage
        starting_stage = st.selectbox(
            "Starting Developmental Stage",
            options=["INFANT", "TODDLER", "CHILD", "ADOLESCENT", "MATURE"],
            index=["INFANT", "TODDLER", "CHILD", "ADOLESCENT", "MATURE"].index(config.mind.starting_stage)
        )
        config.mind.starting_stage = starting_stage
        
        # Step interval
        step_interval = st.slider(
            "Simulation Step Interval (seconds)",
            min_value=0.01,
            max_value=2.0,
            value=config.mind.step_interval,
            step=0.01
        )
        config.mind.step_interval = step_interval
        
        # Development acceleration
        dev_accel = st.slider(
            "Developmental Acceleration Factor",
            min_value=0.1,
            max_value=10.0,
            value=config.mind.development_acceleration,
            step=0.1,
            help="Higher values cause faster developmental progression (1.0 = normal)"
        )
        config.mind.development_acceleration = dev_accel
        
        # Debug mode
        debug_mode = st.checkbox(
            "Debug Mode",
            value=config.development.debug_mode,
            help="Enable more detailed logging and information"
        )
        config.development.debug_mode = debug_mode
        
    with config_tabs[1]:
        st.subheader("Mind Settings")
        
        # Learning rate
        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=config.mind.learning_rate,
            step=0.0001,
            format="%.4f"
        )
        config.mind.learning_rate = learning_rate
        
        # Memory consolidation interval
        memory_interval = st.slider(
            "Memory Consolidation Interval (seconds)",
            min_value=1.0,
            max_value=120.0,
            value=config.mind.memory_consolidation_interval,
            step=1.0
        )
        config.mind.memory_consolidation_interval = memory_interval
        
        # Need update interval
        need_interval = st.slider(
            "Need Update Interval (seconds)",
            min_value=0.1,
            max_value=20.0,
            value=config.mind.need_update_interval,
            step=0.1
        )
        config.mind.need_update_interval = need_interval
        
        # Development check interval
        dev_interval = st.slider(
            "Development Check Interval (seconds)",
            min_value=5.0,
            max_value=300.0,
            value=config.mind.development_check_interval,
            step=5.0
        )
        config.mind.development_check_interval = dev_interval
        
        # Features enabled
        st.subheader("Enabled Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            memory_consolidation = st.checkbox(
                "Memory Consolidation",
                value=config.mind.features_enabled.get("memory_consolidation", True)
            )
            config.mind.features_enabled["memory_consolidation"] = memory_consolidation
            
            emotional_development = st.checkbox(
                "Emotional Development",
                value=config.mind.features_enabled.get("emotional_development", True)
            )
            config.mind.features_enabled["emotional_development"] = emotional_development
            
            belief_formation = st.checkbox(
                "Belief Formation",
                value=config.mind.features_enabled.get("belief_formation", True)
            )
            config.mind.features_enabled["belief_formation"] = belief_formation
            
        with col2:
            language_acquisition = st.checkbox(
                "Language Acquisition",
                value=config.mind.features_enabled.get("language_acquisition", True)
            )
            config.mind.features_enabled["language_acquisition"] = language_acquisition
            
            need_simulation = st.checkbox(
                "Need Simulation",
                value=config.mind.features_enabled.get("need_simulation", True)
            )
            config.mind.features_enabled["need_simulation"] = need_simulation
        
    with config_tabs[2]:
        st.subheader("Neural Network Settings")
        
        # Network configurations
        networks = ["consciousness", "emotions", "perception", "thoughts"]
        
        for network in networks:
            st.subheader(f"{network.capitalize()} Network")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                input_dim = st.number_input(
                    f"{network.capitalize()} Input Dimension",
                    min_value=16,
                    max_value=512,
                    value=config.mind.networks.get(network, {}).get("input_dim", 64),
                    step=16
                )
                config.mind.networks[network]["input_dim"] = input_dim
                
            with col2:
                hidden_dim = st.number_input(
                    f"{network.capitalize()} Hidden Dimension",
                    min_value=32,
                    max_value=1024,
                    value=config.mind.networks.get(network, {}).get("hidden_dim", 128),
                    step=32
                )
                config.mind.networks[network]["hidden_dim"] = hidden_dim
                
            with col3:
                output_dim = st.number_input(
                    f"{network.capitalize()} Output Dimension",
                    min_value=16,
                    max_value=512,
                    value=config.mind.networks.get(network, {}).get("output_dim", 64),
                    step=16
                )
                config.mind.networks[network]["output_dim"] = output_dim
            
    with config_tabs[3]:
        st.subheader("Advanced Settings")
        
        # Model settings
        st.subheader("LLM Model Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=config.model.temperature,
                step=0.1
            )
            config.model.temperature = temperature
            
        with col2:
            max_tokens = st.number_input(
                "Max Tokens (-1 for unlimited)",
                min_value=-1,
                max_value=4096,
                value=config.model.max_tokens,
                step=128
            )
            config.model.max_tokens = max_tokens
            
        # Simulate LLM
        simulate_llm = st.checkbox(
            "Simulate LLM Responses",
            value=config.development.simulate_llm,
            help="Use simulated responses instead of real LLM API calls (for development/testing)"
        )
        config.development.simulate_llm = simulate_llm
        
        # Server URLs
        st.subheader("Server URLs")
        
        llm_server_url = st.text_input(
            "LLM Server URL",
            value=config.server.llm_server_url
        )
        config.server.llm_server_url = llm_server_url
        
        embedding_server_url = st.text_input(
            "Embedding Server URL",
            value=config.server.embedding_server_url
        )
        config.server.embedding_server_url = embedding_server_url
        
        # Export configuration button
        if st.button("Export Configuration"):
            # Create a temporary file
            temp_config_path = "exported_config.yaml"
            config.to_yaml(temp_config_path)
            
            # Read the file content
            with open(temp_config_path, "r") as f:
                config_yaml = f.read()
                
            # Display the content
            st.code(config_yaml, language="yaml")
            
            # Clean up the file
            os.remove(temp_config_path)

def main():
    """Main function for the Streamlit app."""
    try:
        # Set page config
        st.set_page_config(
            page_title="Neural Child Brain Simulation",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Parse command line arguments
        args = parse_arguments()
        
        # Initialize session state - must happen before anything else!
        initialize_session_state()
        
        # Load project configuration
        load_project_configuration(args.config)
        
        # Display the dashboard
        display_dashboard()
        
        # Process message queue from simulation thread
        if st.session_state.running:
            process_message_queue()
    
    except Exception as e:
        # Provide better error handling for debugging
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()