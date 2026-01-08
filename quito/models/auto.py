"""
Auto model classes for QUITO library.

These classes automatically load the appropriate model based on configuration
or model name, following the Hugging Face Transformers pattern.
"""

import os
import json
from typing import Any, Dict, List, Optional, Union, Type
from pathlib import Path
import torch
from omegaconf import OmegaConf
from transformers import PretrainedConfig

from quito.config.model import ModelConfig
from quito.config.base import BaseConfig
from quito.models.base import BaseModel
from quito.config.auto import AutoConfig

MODEL_MAPPING = BaseModel.REGISTRY

class AutoModel:
    """
    Auto model class that automatically loads the appropriate model.

    This class follows the Hugging Face Transformers pattern for automatic
    model loading based on configuration or model name.
    """

    @classmethod
    def from_config(cls, config: Union[ModelConfig, PretrainedConfig], local_rank: int, **kwargs) -> BaseModel:
        """
        Create a model from Quito configuration yaml file, this is the default entrance.

        Args:
            config: Model configuration
            **kwargs: Additional arguments

        Returns:
            Model instance
        """
        model_class = MODEL_MAPPING.get(config.model_name)
        if model_class is None:
            raise ValueError(f"Unknown model type: {config.model_name}")

        return model_class(config, local_rank, **kwargs)

    @classmethod
    def from_pretrained(cls, local_config_path: str = None, local_rank=-1, rank=-1, world_size=-1, **kwargs) -> BaseModel:
        """
        Load a pretrained model from:
        1. A local YAML config file (with checkpoint_path)
        2. HuggingFace model hub (e.g., "amazon/chronos-t5-base")
        3. Local HuggingFace directory (e.g., "/path/to/local/model")
        4. Local QUITO model directory (with config.json and pytorch_model.bin)

        Args:
            local_config_path: Path to Quito YAML config file
            pretrained_model_name_or_path: HuggingFace model identifier or local path
            local_rank: Local rank for distributed training
            rank: Global rank for distributed training
            world_size: World size for distributed training
            **kwargs: Additional arguments passed to model initialization

        Returns:
            Loaded model instance
        """
        # get quito yaml config file
        config = OmegaConf.load(local_config_path)
        _, model_config, training_config = AutoConfig.from_config(config, local_rank=local_rank, rank=rank, world_size=world_size)


        model = cls.from_config(model_config, local_rank=local_rank, **kwargs)

        if local_config_path:
            # Load from local config file, need to first load config then init model.
            config = OmegaConf.load(local_config_path)
            _, model_config, training_config = AutoConfig.from_config(config, local_rank=local_rank, rank=rank, world_size=world_size)
            model = cls.from_config(model_config, local_rank=local_rank, **kwargs)
            checkpoint_path = training_config.checkpoint_path
            if checkpoint_path:
                model.load(checkpoint_path)

            return model

        elif pretrained_model_name_or_path:
            # Try to load from HuggingFace hub or local directory
            return cls._from_pretrained_hf_or_local(pretrained_model_name_or_path, local_rank=local_rank, **kwargs)
        else:
            raise ValueError(
                "Either 'local_config_path' or 'pretrained_model_name_or_path' must be provided"
            )

    @classmethod
    def _from_pretrained_hf_or_local(cls, pretrained_model_name_or_path: str, local_rank: int = -1, **kwargs) -> BaseModel:
        """
        Load a pretrained model from HuggingFace hub or local directory.

        Args:
            pretrained_model_name_or_path: Model identifier (hub) or local path
            local_rank: Local rank for device placement
            **kwargs: Additional arguments

        Returns:
            Loaded model instance
        """
        from pathlib import Path
        import os

        model_path = Path(pretrained_model_name_or_path)

        # Check if it's a local path
        is_local_path = model_path.exists() and model_path.is_dir()

        # Check if it's a HuggingFace model (has config.json with model_type)
        is_hf_model = False
        if is_local_path:
            config_file = model_path / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        hf_config = json.load(f)
                    # Check if it's a HuggingFace config (has model_type)
                    if 'model_type' in hf_config:
                        is_hf_model = True
                except:
                    pass

        # Try to load as HuggingFace model first
        if is_hf_model or not is_local_path:
            # Try loading as HuggingFace model
            try:
                return cls._from_pretrained_huggingface(
                    pretrained_model_name_or_path,
                    local_rank=local_rank,
                    **kwargs
                )
            except Exception as e:
                # If it fails and it's a local path, try QUITO format
                if is_local_path:
                    return cls._from_pretrained_local(pretrained_model_name_or_path, local_rank=local_rank, **kwargs)
                else:
                    raise ValueError(
                        f"Failed to load model from HuggingFace hub '{pretrained_model_name_or_path}': {e}\n"
                        "Make sure the model identifier is correct and the model exists on HuggingFace Hub."
                    ) from e
        else:
            # Try loading as QUITO model format
            return cls._from_pretrained_local(pretrained_model_name_or_path, local_rank=local_rank, **kwargs)

    @classmethod
    def _from_pretrained_huggingface(cls, pretrained_model_name_or_path: str, local_rank: int = -1, **kwargs) -> BaseModel:
        """
        Load a model from HuggingFace model hub or local HuggingFace directory.

        This method attempts to load the model using HuggingFace's AutoModel,
        which works for models that are compatible with transformers library.
        """
        from transformers import AutoConfig as HFAutoConfig
        from pathlib import Path

        # Load HuggingFace config to determine model type
        try:
            hf_config = HFAutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        except Exception as e:
            raise ValueError(
                f"Could not load HuggingFace config from '{pretrained_model_name_or_path}': {e}\n"
                "Make sure the path is correct and the model is available."
            ) from e

        # Try to infer QUITO model type from HuggingFace model type
        # This is a heuristic - we check common patterns
        model_type_map = {
            'chronos': ModelType.CHRONOS,
            't5': ModelType.CHRONOS,  # Chronos uses T5
            'moirai': ModelType.MORIAI,
            'patchtst': ModelType.PATCHTST,
        }

        hf_model_type = getattr(hf_config, 'model_type', '').lower()
        quito_model_type = None

        # Check model type
        for key, model_type in model_type_map.items():
            if key in hf_model_type or key in pretrained_model_name_or_path.lower():
                quito_model_type = model_type
                break

        # If we can't determine, try to use HuggingFaceModel wrapper
        if quito_model_type is None:
            # Use generic HuggingFace wrapper
            from quito.config.model import HuggingFaceModelConfig
            model_config = HuggingFaceModelConfig(
                model_name=ModelType.HUGGINGFACE,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                seq_len=kwargs.get('seq_len', 512),
                forecast_horizon=kwargs.get('forecast_horizon', 192),
                trust_remote_code=kwargs.get('trust_remote_code', True),
            )
            return cls.from_config(model_config, local_rank=local_rank, **kwargs)

        # Load the specific model type
        model_class = MODEL_MAPPING.get(quito_model_type)
        if model_class is None:
            raise ValueError(f"Model type {quito_model_type} not found in MODEL_MAPPING")

        # Create config for the specific model type
        # We need to infer config from HuggingFace config or use defaults
        from quito.config.model import ModelConfig

        # Get default config for this model type
        if quito_model_type == ModelType.CHRONOS:
            from quito.config.model import ChronosModelConfig
            model_config = ChronosModelConfig(
                model_name=ModelType.CHRONOS,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                seq_len=kwargs.get('seq_len', 512),
                forecast_horizon=kwargs.get('forecast_horizon', 192),
            )
        elif quito_model_type == ModelType.MORIAI:
            from quito.config.model import MoriaiModelConfig
            model_config = MoriaiModelConfig(
                model_name=ModelType.MORIAI,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                seq_len=kwargs.get('seq_len', 512),
                forecast_horizon=kwargs.get('forecast_horizon', 192),
            )
        else:
            # For other types, create a basic config
            model_config = ModelConfig(
                model_name=quito_model_type,
                seq_len=kwargs.get('seq_len', 512),
                forecast_horizon=kwargs.get('forecast_horizon', 192),
            )
            if hasattr(model_config, 'pretrained_model_name_or_path'):
                model_config.pretrained_model_name_or_path = pretrained_model_name_or_path

        return model_class(model_config, local_rank=local_rank, **kwargs)

    @classmethod
    def _from_pretrained_remote(cls, model_name: str, **kwargs) -> BaseModel:
        """
        Load a model from remote source (model hub, etc.).

        Args:
            model_name: Model name
            **kwargs: Additional arguments

        Returns:
            Loaded model instance
        """
        # TODO: we need able to load from huggingface repo
        pass

    @classmethod
    def _from_local_checkpoint(cls, checkpoint_path: str, **kwargs):

        pass

    @classmethod
    def _from_pretrained_local(cls, model_path: str, local_rank: int = -1, **kwargs) -> BaseModel:
        """
        Load a QUITO model from a local directory.

        Expected directory structure:
        - config.json (QUITO ModelConfig)
        - pytorch_model.bin (model state dict)
        - model_info.json (optional, metadata)

        Args:
            model_path: Path to the model directory
            local_rank: Local rank for device placement
            **kwargs: Additional arguments

        Returns:
            Loaded model instance
        """
        from pathlib import Path

        model_path = Path(model_path)
        if not model_path.exists() or not model_path.is_dir():
            raise ValueError(f"Model path does not exist or is not a directory: {model_path}")

        # Load QUITO config
        config_file = model_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_file}\n"
                "Expected QUITO model format with config.json"
            )

        try:
            # Load config as ModelConfig
            with open(config_file, 'r') as f:
                config_dict = json.load(f)

            # Determine model type and load appropriate config class
            model_name = config_dict.get('model_name')
            if model_name:
                # Convert string to ModelType if needed
                if isinstance(model_name, str):
                    try:
                        model_type = ModelType(model_name)
                    except ValueError:
                        raise ValueError(
                            f"Unknown model type: {model_name}\n"
                            f"Supported types: {[mt.value for mt in ModelType]}"
                        )
                else:
                    model_type = model_name

                # Get the appropriate config class for this model type
                from quito.config.auto import MODEL_CONFIG_MAPPING
                config_class = MODEL_CONFIG_MAPPING.get(model_type)
                if config_class is None:
                    raise ValueError(f"No config class found for model type: {model_type}")

                # Create config instance from dict
                model_config = config_class.from_dict(config_dict)
            else:
                # Fallback to generic ModelConfig
                model_config = ModelConfig.from_dict(config_dict)
        except Exception as e:
            raise ValueError(
                f"Failed to load model config from {config_file}: {e}\n"
                "Make sure the config.json is in QUITO ModelConfig format."
            ) from e

        # Create model instance
        model = cls.from_config(model_config, local_rank=local_rank, **kwargs)

        # Load model weights
        model_file = model_path / "pytorch_model.bin"
        if model_file.exists():
            try:
                device = f"cuda:{local_rank}" if local_rank >= 0 and torch.cuda.is_available() else "cpu"
                state_dict = torch.load(model_file, map_location=device)
                model.load_state_dict(state_dict)
            except Exception as e:
                raise ValueError(
                    f"Failed to load model weights from {model_file}: {e}\n"
                    "Make sure the pytorch_model.bin contains a valid state dict."
                ) from e
        else:
            # Try alternative checkpoint format
            checkpoint_file = model_path / "checkpoint.pt"
            if checkpoint_file.exists():
                model.load(str(checkpoint_file))
            else:
                import logging
                logging.warning(
                    f"No model weights found in {model_path}. "
                    "Model will be initialized with random weights."
                )

        return model

    @classmethod
    def _from_pretrained_hf_or_local(cls, pretrained_model_name_or_path: str, local_rank: int = -1, **kwargs) -> BaseModel:
        """
        Load a pretrained model from HuggingFace hub or local directory.

        Args:
            pretrained_model_name_or_path: Model identifier (hub) or local path
            local_rank: Local rank for device placement
            **kwargs: Additional arguments

        Returns:
            Loaded model instance
        """
        from pathlib import Path
        import os

        model_path = Path(pretrained_model_name_or_path)

        # Check if it's a local path
        is_local_path = model_path.exists() and model_path.is_dir()

        # Check if it's a HuggingFace model (has config.json with model_type)
        is_hf_model = False
        if is_local_path:
            config_file = model_path / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        hf_config = json.load(f)
                    # Check if it's a HuggingFace config (has model_type)
                    if 'model_type' in hf_config:
                        is_hf_model = True
                except:
                    pass

        # Try to load as HuggingFace model first
        if is_hf_model or not is_local_path:
            # Try loading as HuggingFace model
            try:
                return cls._from_pretrained_huggingface(
                    pretrained_model_name_or_path,
                    local_rank=local_rank,
                    **kwargs
                )
            except Exception as e:
                # If it fails and it's a local path, try QUITO format
                if is_local_path:
                    return cls._from_pretrained_local(pretrained_model_name_or_path, local_rank=local_rank, **kwargs)
                else:
                    raise ValueError(
                        f"Failed to load model from HuggingFace hub '{pretrained_model_name_or_path}': {e}\n"
                        "Make sure the model identifier is correct and the model exists on HuggingFace Hub."
                    ) from e
        else:
            # Try loading as QUITO model format
            return cls._from_pretrained_local(pretrained_model_name_or_path, local_rank=local_rank, **kwargs)

    @classmethod
    def _from_pretrained_huggingface(cls, pretrained_model_name_or_path: str, local_rank: int = -1, **kwargs) -> BaseModel:
        """
        Load a model from HuggingFace model hub or local HuggingFace directory.

        This method attempts to load the model using HuggingFace's AutoModel,
        which works for models that are compatible with transformers library.
        """
        from transformers import AutoConfig as HFAutoConfig
        from pathlib import Path

        # Load HuggingFace config to determine model type
        try:
            hf_config = HFAutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        except Exception as e:
            raise ValueError(
                f"Could not load HuggingFace config from '{pretrained_model_name_or_path}': {e}\n"
                "Make sure the path is correct and the model is available."
            ) from e

        # Try to infer QUITO model type from HuggingFace model type
        # This is a heuristic - we check common patterns
        model_type_map = {
            'chronos': ModelType.CHRONOS,
            't5': ModelType.CHRONOS,  # Chronos uses T5
            'moirai': ModelType.MORIAI,
            'patchtst': ModelType.PATCHTST,
        }

        hf_model_type = getattr(hf_config, 'model_type', '').lower()
        quito_model_type = None

        # Check model type
        for key, model_type in model_type_map.items():
            if key in hf_model_type or key in pretrained_model_name_or_path.lower():
                quito_model_type = model_type
                break

        # If we can't determine, try to use HuggingFaceModel wrapper
        if quito_model_type is None:
            # Use generic HuggingFace wrapper
            from quito.config.model import HuggingFaceModelConfig
            model_config = HuggingFaceModelConfig(
                model_name=ModelType.HUGGINGFACE,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                seq_len=kwargs.get('seq_len', 512),
                forecast_horizon=kwargs.get('forecast_horizon', 192),
                trust_remote_code=kwargs.get('trust_remote_code', True),
            )
            return cls.from_config(model_config, local_rank=local_rank, **kwargs)

        # Load the specific model type
        model_class = MODEL_MAPPING.get(quito_model_type)
        if model_class is None:
            raise ValueError(f"Model type {quito_model_type} not found in MODEL_MAPPING")

        # Create config for the specific model type
        # We need to infer config from HuggingFace config or use defaults
        from quito.config.model import ModelConfig

        # Get default config for this model type
        if quito_model_type == ModelType.CHRONOS:
            from quito.config.model import ChronosModelConfig
            model_config = ChronosModelConfig(
                model_name=ModelType.CHRONOS,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                seq_len=kwargs.get('seq_len', 512),
                forecast_horizon=kwargs.get('forecast_horizon', 192),
            )
        elif quito_model_type == ModelType.MORIAI:
            from quito.config.model import MoriaiModelConfig
            model_config = MoriaiModelConfig(
                model_name=ModelType.MORIAI,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                seq_len=kwargs.get('seq_len', 512),
                forecast_horizon=kwargs.get('forecast_horizon', 192),
            )
        else:
            # For other types, create a basic config
            model_config = ModelConfig(
                model_name=quito_model_type,
                seq_len=kwargs.get('seq_len', 512),
                forecast_horizon=kwargs.get('forecast_horizon', 192),
            )
            if hasattr(model_config, 'pretrained_model_name_or_path'):
                model_config.pretrained_model_name_or_path = pretrained_model_name_or_path

        return model_class(model_config, local_rank=local_rank, **kwargs)

    @classmethod
    def _from_pretrained_remote(cls, model_name: str, **kwargs) -> BaseModel:
        """
        Load a model from remote source (model hub, etc.).

        Args:
            model_name: Model name
            **kwargs: Additional arguments

        Returns:
            Loaded model instance
        """
        # TODO: we need able to load from huggingface repo
        pass

    @classmethod
    def _from_local_checkpoint(cls, checkpoint_path: str, **kwargs):

        pass

    @classmethod
    def _from_pretrained_local(cls, model_path: str, local_rank: int = -1, **kwargs) -> BaseModel:
        """
        Load a QUITO model from a local directory.

        Expected directory structure:
        - config.json (QUITO ModelConfig)
        - pytorch_model.bin (model state dict)
        - model_info.json (optional, metadata)

        Args:
            model_path: Path to the model directory
            local_rank: Local rank for device placement
            **kwargs: Additional arguments

        Returns:
            Loaded model instance
        """
        from pathlib import Path

        model_path = Path(model_path)
        if not model_path.exists() or not model_path.is_dir():
            raise ValueError(f"Model path does not exist or is not a directory: {model_path}")

        # Load QUITO config
        config_file = model_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_file}\n"
                "Expected QUITO model format with config.json"
            )

        try:
            # Load config as ModelConfig
            with open(config_file, 'r') as f:
                config_dict = json.load(f)

            # Determine model type and load appropriate config class
            model_name = config_dict.get('model_name')
            if model_name:
                # Convert string to ModelType if needed
                if isinstance(model_name, str):
                    try:
                        model_type = ModelType(model_name)
                    except ValueError:
                        raise ValueError(
                            f"Unknown model type: {model_name}\n"
                            f"Supported types: {[mt.value for mt in ModelType]}"
                        )
                else:
                    model_type = model_name

                # Get the appropriate config class for this model type
                from quito.config.auto import MODEL_CONFIG_MAPPING
                config_class = MODEL_CONFIG_MAPPING.get(model_type)
                if config_class is None:
                    raise ValueError(f"No config class found for model type: {model_type}")

                # Create config instance from dict
                model_config = config_class.from_dict(config_dict)
            else:
                # Fallback to generic ModelConfig
                model_config = ModelConfig.from_dict(config_dict)
        except Exception as e:
            raise ValueError(
                f"Failed to load model config from {config_file}: {e}\n"
                "Make sure the config.json is in QUITO ModelConfig format."
            ) from e

        # Create model instance
        model = cls.from_config(model_config, local_rank=local_rank, **kwargs)

        # Load model weights
        model_file = model_path / "pytorch_model.bin"
        if model_file.exists():
            try:
                device = f"cuda:{local_rank}" if local_rank >= 0 and torch.cuda.is_available() else "cpu"
                state_dict = torch.load(model_file, map_location=device)
                model.load_state_dict(state_dict)
            except Exception as e:
                raise ValueError(
                    f"Failed to load model weights from {model_file}: {e}\n"
                    "Make sure the pytorch_model.bin contains a valid state dict."
                ) from e
        else:
            # Try alternative checkpoint format
            checkpoint_file = model_path / "checkpoint.pt"
            if checkpoint_file.exists():
                model.load(str(checkpoint_file))
            else:
                import logging
                logging.warning(
                    f"No model weights found in {model_path}. "
                    "Model will be initialized with random weights."
                )

        return model

    @classmethod
    def register(cls, model_type: str, model_class: Type[BaseModel]):
        """
        Register a new model type.

        Args:
            model_type: Model type enum
            model_class: Model class
        """
        MODEL_MAPPING[model_type] = model_class

    @classmethod
    def list_models(cls) -> List[str]:
        """
        List all available model types.

        Returns:
            List of model type names
        """
        return [model_type.value for model_type in MODEL_MAPPING]

