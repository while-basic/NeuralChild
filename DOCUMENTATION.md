# Neural Child Project Documentation

## Overview

The Neural Child project is a psychological brain simulation that models the development of a child's mind using neural networks and large language models. It simulates various cognitive processes including consciousness, emotions, perception, and thoughts, with a "Mother" LLM providing nurturing interactions.

## System Architecture

### Core Components

1. **Mind**: The central component that simulates a developing mind
   - Manages developmental stages (Infant, Toddler, Child, Adolescent, Mature)
   - Coordinates neural networks for different cognitive functions
   - Processes inputs and generates outputs

2. **Neural Networks**: Specialized networks for different cognitive functions
   - **ConsciousnessNetwork**: Simulates awareness and self-model
   - **EmotionsNetwork**: Processes and generates emotional responses
   - **PerceptionNetwork**: Handles visual and auditory processing
   - **ThoughtsNetwork**: Generates thoughts and cognitive processes

3. **Mother LLM**: Simulates a caregiver using large language models
   - Observes the child's state
   - Provides nurturing responses
   - Guides development through interactions

4. **Message Bus**: Communication system between components
   - Facilitates message passing between networks
   - Implements publish-subscribe pattern
   - Supports message filtering and prioritization

5. **Dashboard**: Interactive visualization of the mind's development
   - Real-time monitoring of mind state
   - Visualization of emotions, development, and memory
   - Controls for simulation parameters

### Data Flow

1. Environmental inputs are processed by the Mind
2. The Mind distributes inputs to appropriate neural networks
3. Networks process inputs and communicate via the Message Bus
4. The Mother LLM observes the Mind's state and provides responses
5. The Dashboard visualizes the current state and history

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- Dash and related packages for the dashboard

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neuralchild.git
cd neuralchild
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
python run_dashboard.py
```

## Using the Dashboard

### Main Interface

The dashboard provides a comprehensive view of the Neural Child's development:

- **Status Panel**: Shows the current developmental stage, energy level, and mood
- **Network Outputs**: Displays the current outputs from each neural network
- **Development Graph**: Visualizes developmental progress over time
- **Emotions Graph**: Shows emotional state changes over time
- **Memory Graph**: Displays memory formation and retention
- **Milestones**: Lists developmental milestones achieved
- **Beliefs**: Shows the child's developing belief system
- **Needs**: Displays the child's current needs

### Controls

- **Start/Stop**: Control the simulation
- **Save**: Save the current state of all models
- **Configuration**: Adjust simulation parameters
  - Step interval: Time between simulation steps
  - Save interval: Steps between automatic saves
  - Checkpoint count: Number of model checkpoints to keep
  - Save directory: Location for saved models
  - Auto backup: Enable/disable automatic backups

### Interacting with the Child

The dashboard allows for simulated interactions with the child:

1. Observe the child's current state in the status panel
2. See the Mother's responses to the child
3. Monitor how interactions affect the child's development

## Development Guide

### Project Structure

- `mind/`: Core mind simulation components
  - `networks/`: Neural network implementations
  - `mind_core.py`: Main Mind class implementation
- `mother/`: Mother LLM implementation
- `core/`: Core neural network and schema implementations
- `communication/`: Message bus and communication components
- `utils/`: Utility functions
- `tests/`: Test suite
- `neural-child-dashboard.py`: Dashboard implementation

### Adding a New Network

1. Create a new network class in `mind/networks/`
2. Inherit from `NeuralNetwork` in `core/neural_network.py`
3. Implement required methods:
   - `forward()`: Forward pass of the network
   - `process_message()`: Handle messages from other networks
   - `generate_text_output()`: Generate human-readable output
4. Add the network to the Mind class in `mind_core.py`
5. Update visualization components in the dashboard

### Extending the Mother LLM

1. Modify `mother/mother_llm.py` to add new capabilities
2. Update the prompt templates in `mother/prompts.py`
3. Add new interaction patterns in `mother_llm.py`

## Testing

The project includes a comprehensive test suite:

```bash
python run_tests.py
```

Tests cover:
- Mind initialization and processing
- Neural network functionality
- Mother LLM interactions
- Dashboard components

## Troubleshooting

### Common Issues

1. **Port Conflict**: If the dashboard fails to start due to a port conflict, modify the port in `neural-child-dashboard.py`:
   ```python
   app.run_server(debug=True, port=8051)  # Change port as needed
   ```

2. **Memory Usage**: The simulation can be memory-intensive. If you encounter memory issues:
   - Reduce the size of neural networks
   - Limit the history length in the dashboard
   - Run with a smaller step interval

3. **LLM API Issues**: If using external LLM APIs:
   - Check API keys and rate limits
   - Enable simulation mode in config for testing

## Advanced Configuration

### Configuration File

The `config.yaml` file allows for detailed configuration:

```yaml
development:
  initial_stage: "INFANT"
  growth_rate: 0.01
  simulate_llm: true

networks:
  consciousness:
    input_dim: 64
    hidden_dim: 128
    output_dim: 64
  emotions:
    input_dim: 64
    hidden_dim: 128
    output_dim: 64
  # Additional network configurations...

mother:
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 150
```

### Custom Neural Networks

To create custom neural networks with specialized architectures:

1. Extend the `NeuralNetwork` base class
2. Implement custom layers and connections
3. Override growth and pruning methods
4. Register the network with the Mind

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Run tests to ensure functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 