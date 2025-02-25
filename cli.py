"""Command-line interface for the NeuralChild project."""

import argparse
import time
from typing import Optional

from neuralchild.config import load_config, Config
from neuralchild.mind.mind_core import Mind
from neuralchild.mother.mother_llm import MotherLLM

def run(config_path: Optional[str] = None) -> None:
    """Run the NeuralChild simulation."""
    # Load configuration
    config = load_config(config_path) if config_path else Config()
    
    # Initialize components
    mind = Mind()
    mother = MotherLLM()
    
    # TODO: Initialize visualization if enabled
    
    # Main simulation loop
    try:
        print("🧠 Starting NeuralChild simulation...")
        print("Press Ctrl+C to stop")
        
        while True:
            # Advance mind simulation
            mind.step()
            
            # Get observable state and generate mother response
            response = mother.observe_and_respond(mind)
            if response:
                print(f"\n👩 Mother: {response.response}")
            
            # Wait for next step
            time.sleep(config.mind.step_interval)
            
    except KeyboardInterrupt:
        print("\nStopping NeuralChild simulation...")

def main() -> None:
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="NeuralChild: A psychological brain simulation")
    parser.add_argument("--config", "-c", type=str, help="Path to the configuration file")
    
    args = parser.parse_args()
    run(args.config)

if __name__ == "__main__":
    main()