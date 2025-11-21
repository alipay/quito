"""
Base configuration class for QUITO library.
"""

import os
import json
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml
from omegaconf import OmegaConf


@dataclass
class BaseConfig:
    """
    Base configuration class that provides common functionality for all configs.
    
    This class provides methods for saving/loading configurations and
    validation of configuration parameters.
    """
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self):
        """Validate the configuration parameters."""
        # Override in subclasses
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_omega_conf(self) -> str:
        """Convert configuration to YAML string."""
        # create omegaconf object from dict
        conf = OmegaConf.create(self.to_dict())
        
        return conf
    
    def save(self, path: Union[str, Path], format: str = "yaml"):
        """
        Save configuration to file.
        
        Args:
            path: Path to save the configuration
            format: Format to save in ("json", "yaml", "yml")
        """
        if format.lower() == "json":
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif format.lower() in ["yaml", "yml"]:
            OmegaConf.save(self.to_omega_conf(), path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> "BaseConfig":
        """Create configuration from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> "BaseConfig":
        """Create configuration from YAML string."""
        config_dict = yaml.safe_load(yaml_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "BaseConfig":
        """
        Load configuration from file.
        
        Args:
            path: Path to the configuration file
            
        Returns:
            Configuration instance
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r") as f:
            if path.suffix.lower() == ".json":
                config_dict = json.load(f)
            elif path.suffix.lower() in [".yaml", ".yml"]:
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        # Re-validate after update
        self.validate()
    
    def copy(self) -> "BaseConfig":
        """Create a copy of the configuration."""
        return self.__class__(**self.to_dict())
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"{self.__class__.__name__}({self.to_dict()})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return self.__str__() 