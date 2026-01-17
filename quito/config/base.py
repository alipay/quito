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
    Base configuration class providing common functionality for all QuitoBench configs.
    
    This abstract base class provides a foundation for all configuration objects in
    QuitoBench. It includes methods for serialization (JSON, YAML), validation,
    loading/saving, and updating configurations. Subclasses should override the
    validate() method to implement custom validation logic.
    
    Attributes:
        All attributes are defined by subclasses using dataclass fields.
        
    Example:
        >>> @dataclass
        ... class MyConfig(BaseConfig):
        ...     learning_rate: float = 0.001
        ...     def validate(self):
        ...         if self.learning_rate <= 0:
        ...             raise ValueError("learning_rate must be positive")
        >>> config = MyConfig(learning_rate=0.01)
        >>> config.save("config.yaml")
    """
    
    def __post_init__(self):
        """
        Validate configuration after dataclass initialization.
        
        Automatically called after object creation to ensure configuration
        parameters are valid. Calls the validate() method which should be
        overridden in subclasses.
        """
        self.validate()
    
    def validate(self):
        """
        Validate the configuration parameters.
        
        This method should be overridden in subclasses to implement custom
        validation logic. Raises ValueError or other exceptions if validation fails.
        
        Raises:
            ValueError: If any configuration parameter is invalid.
            
        Note:
            This is a no-op in the base class. Subclasses must implement
            their own validation logic.
        """
        # Override in subclasses
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary containing all configuration fields
                and their values.
                
        Example:
            >>> config = MyConfig(learning_rate=0.01, batch_size=32)
            >>> config_dict = config.to_dict()
            >>> # {'learning_rate': 0.01, 'batch_size': 32}
        """
        return asdict(self)
    
    def to_json(self) -> str:
        """
        Convert configuration to a JSON string.
        
        Returns:
            str: JSON-formatted string representation of the configuration.
            
        Example:
            >>> config = MyConfig(learning_rate=0.01)
            >>> json_str = config.to_json()
        """
        return json.dumps(self.to_dict(), indent=2)
    
    def to_omega_conf(self) -> str:
        """
        Convert configuration to an OmegaConf object.
        
        OmegaConf is used internally for YAML handling and provides advanced
        features like variable interpolation and hierarchical configuration.
        
        Returns:
            OmegaConf.DictConfig: OmegaConf configuration object.
            
        Example:
            >>> config = MyConfig(learning_rate=0.01)
            >>> omega_conf = config.to_omega_conf()
            >>> # Can be used with OmegaConf.save() for YAML output
        """
        # create omegaconf object from dict
        conf = OmegaConf.create(self.to_dict())
        
        return conf
    
    def save(self, path: Union[str, Path], format: str = "yaml"):
        """
        Save configuration to a file in the specified format.
        
        Supports JSON and YAML formats. Creates parent directories if they don't exist.
        
        Args:
            path (Union[str, Path]): File path where configuration will be saved.
            format (str, optional): File format. Options: "json", "yaml", "yml".
                Defaults to "yaml".
                
        Raises:
            ValueError: If an unsupported format is specified.
            IOError: If file cannot be written.
            
        Example:
            >>> config = MyConfig(learning_rate=0.01)
            >>> config.save("configs/my_config.yaml")
            >>> config.save("configs/my_config.json", format="json")
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
        """
        Create a configuration instance from a dictionary.
        
        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration
                field names and values.
        
        Returns:
            BaseConfig: New configuration instance.
            
        Example:
            >>> config_dict = {'learning_rate': 0.01, 'batch_size': 32}
            >>> config = MyConfig.from_dict(config_dict)
        """
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> "BaseConfig":
        """
        Create a configuration instance from a JSON string.
        
        Args:
            json_str (str): JSON-formatted string containing configuration.
        
        Returns:
            BaseConfig: New configuration instance.
            
        Raises:
            json.JSONDecodeError: If JSON string is malformed.
            
        Example:
            >>> json_str = '{"learning_rate": 0.01, "batch_size": 32}'
            >>> config = MyConfig.from_json(json_str)
        """
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> "BaseConfig":
        """
        Create a configuration instance from a YAML string.
        
        Args:
            yaml_str (str): YAML-formatted string containing configuration.
        
        Returns:
            BaseConfig: New configuration instance.
            
        Raises:
            yaml.YAMLError: If YAML string is malformed.
            
        Example:
            >>> yaml_str = "learning_rate: 0.01\\nbatch_size: 32"
            >>> config = MyConfig.from_yaml(yaml_str)
        """
        config_dict = yaml.safe_load(yaml_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "BaseConfig":
        """
        Load configuration from a file (JSON or YAML).
        
        Automatically detects file format based on file extension (.json, .yaml, .yml).
        
        Args:
            path (Union[str, Path]): Path to the configuration file.
        
        Returns:
            BaseConfig: New configuration instance loaded from file.
            
        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the file format is unsupported.
            json.JSONDecodeError: If JSON file is malformed.
            yaml.YAMLError: If YAML file is malformed.
            
        Example:
            >>> config = MyConfig.from_file("configs/training_config.yaml")
            >>> config = MyConfig.from_file("configs/training_config.json")
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
        """
        Update configuration parameters with new values.
        
        Updates specified fields and re-validates the configuration. Only
        existing fields can be updated; unknown fields raise an error.
        
        Args:
            **kwargs: Keyword arguments mapping field names to new values.
        
        Raises:
            ValueError: If an unknown configuration parameter is specified,
                or if validation fails after update.
                
        Example:
            >>> config = MyConfig(learning_rate=0.01)
            >>> config.update(learning_rate=0.001, batch_size=64)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        # Re-validate after update
        self.validate()
    
    def copy(self) -> "BaseConfig":
        """
        Create a deep copy of the configuration.
        
        Returns:
            BaseConfig: A new configuration instance with identical values.
            
        Example:
            >>> config = MyConfig(learning_rate=0.01)
            >>> config_copy = config.copy()
            >>> config_copy.update(learning_rate=0.001)  # Original unchanged
        """
        return self.__class__(**self.to_dict())
    
    def __str__(self) -> str:
        """
        String representation of the configuration.
        
        Returns:
            str: Human-readable string showing class name and all fields.
        """
        return f"{self.__class__.__name__}({self.to_dict()})"
    
    def __repr__(self) -> str:
        """
        Detailed string representation of the configuration.
        
        Returns:
            str: Same as __str__ for consistency.
        """
        return self.__str__() 