"""Configuration management using Pydantic models."""

from pydantic import BaseModel, Field
from typing import Optional
import yaml
import os

class ServerConfig(BaseModel):
    """Configuration for external servers."""
    llm_server_url: str = "http://192.168.2.12:1234/v1/chat/completions"
    embedding_server_url: str = "http://192.168.2.12:1234/v1/embeddings"
    obsidian_api_url: Optional[str] = None

class ModelConfig(BaseModel):
    """Configuration for models."""
    llm_model: str = "qwen2.5-7b-instruct"
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = -1  # -1 for unlimited

class VisualizationConfig(BaseModel):
    """Configuration for visualization."""
    enabled: bool = True
    update_interval: float = 1.0  # seconds

class MindConfig(BaseModel):
    """Configuration for the mind simulation."""
    learning_rate: float = 0.001
    step_interval: float = 0.1  # seconds

class Config(BaseModel):
    """Main configuration for the NeuralChild project."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    mind: MindConfig = Field(default_factory=MindConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        if not os.path.exists(path):
            return cls()
            
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        return cls.model_validate(config_dict)
        
    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f)

# Global configuration instance
config = Config()

def load_config(path: str = "config.yaml") -> Config:
    """Load configuration from a YAML file."""
    global config
    config = Config.from_yaml(path)
    return config