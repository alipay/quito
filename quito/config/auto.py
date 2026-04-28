import logging
from omegaconf import DictConfig, OmegaConf
from typing import Union
from transformers import AutoConfig as HFAutoConfig
from transformers import PretrainedConfig

from quito.config.model import ModelConfig
from quito.config.training import TrainerConfig
from quito.config.data import DataConfig

MODEL_CONFIG_MAPPING = ModelConfig.REGISTRY
TRAINER_CONFIG_MAPPING = TrainerConfig.REGISTRY


class AutoConfig:
    """
    Automatic configuration factory for QuitoBench.

    This class automatically creates and configures ModelConfig, DataConfig, and
    TrainerConfig objects from a main YAML configuration file. It handles both
    local QuitoBench model configs and HuggingFace pretrained model configs.

    The class intelligently determines the appropriate config classes based on
    model names and trainer names, enabling automatic discovery and instantiation.

    Example:
        >>> from quito.config.auto import AutoConfig
        >>> data_config, model_config, training_config = AutoConfig.from_config(
        ...     "configs/pretrain/patchtst/config.yaml",
        ...     local_rank=0,
        ...     rank=0,
        ...     world_size=4
        ... )
    """

    @classmethod
    def from_config(cls, config: Union[DictConfig, str], local_rank=-1, rank=-1, world_size=1, **kwargs):
        """
        Create model, data, and training configurations from a main config file.

        Parses a YAML configuration file (or DictConfig object) and automatically
        creates the appropriate configuration objects for data loading, model
        architecture, and training setup.

        Args:
            config (Union[DictConfig, str]): Path to YAML config file or OmegaConf
                DictConfig object containing all configuration sections.
            local_rank (int, optional): Local rank for distributed training.
                Defaults to -1.
            rank (int, optional): Global rank for distributed training.
                Defaults to -1.
            world_size (int, optional): Total number of processes in distributed
                training. Defaults to 1.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            Tuple[DataConfig, Union[ModelConfig, PretrainedConfig], TrainerConfig]:
                A tuple containing:
                - data_config: Data configuration for dataset loading
                - model_config: Model configuration (local or HuggingFace)
                - training_config: Training configuration with distributed settings

        Raises:
            FileNotFoundError: If config file path doesn't exist.
            KeyError: If required configuration sections are missing.
            ValueError: If model or trainer names are not recognized.

        Example:
            >>> data_cfg, model_cfg, train_cfg = AutoConfig.from_config(
            ...     "configs/evaluate/chronos/config.yaml",
            ...     local_rank=0
            ... )
        """
        if isinstance(config, str):
            config = OmegaConf.load(config)

        # get model config
        data_config = cls._get_data_config(config)
        model_config = cls._get_model_config(config)
        training_config = cls._get_training_config(config=config, rank=rank, local_rank=local_rank,
                                                   world_size=world_size)
        # set checkpoint path to model's actual checkpoint path
        training_config.checkpoint_path = model_config.checkpoint_path

        return data_config, model_config, training_config

    @staticmethod
    def _get_model_config(config: DictConfig) -> Union[ModelConfig, PretrainedConfig]:
        """
        Extract and create model configuration from main config.

        Determines whether to use a local QuitoBench ModelConfig or load a
        HuggingFace PretrainedConfig based on the presence of
        'pretrained_model_name_or_path'. Handles special cases like TiRex.

        Args:
            config (DictConfig): Main configuration dictionary containing
                'model' and 'data.common' sections.

        Returns:
            Union[ModelConfig, PretrainedConfig]: Model configuration object
                with seq_len, forecast_horizon, and decoder_label_len set
                from data configuration.

        Raises:
            KeyError: If model configuration is missing required fields.
            ValueError: If model_name is not recognized in MODEL_CONFIG_MAPPING.
        """
        # get model config
        model_config = config.model
        data_config = config.data.common
        checkpoint_config = config.resume
        # check for pretrained config (huggingface PretrainConfig) or local config (ModelConfig)
        if 'pretrained_model_name_or_path' in model_config and model_config.pretrained_model_name_or_path:
            if isinstance(model_config.pretrained_model_name_or_path, list):
                raise ValueError('pretrained_model_name_or_path should be a string, not a list, '
                                 'multiple checkpoints are only supported using quito checkpoints')
            # this will load pretrained model config.json from huggingface or local path
            if model_config.model_name == 'TiRex':
                # for tirex, only checkpoint path is provided, the config is inside the construct a empty config file
                model_config_name = f'TiRexModelConfig'
                model_config_cls = MODEL_CONFIG_MAPPING[model_config_name]
                curr_model_config = model_config_cls(**model_config)
                logging.info(f'creating TiRex model config from local {curr_model_config.__class__.__name__}')
            else:
                curr_model_config = HFAutoConfig.from_pretrained(model_config.pretrained_model_name_or_path)
                logging.info(f'creating model config from huggingface using {curr_model_config.__class__.__name__}')

            curr_model_config.checkpoint_path = model_config.pretrained_model_name_or_path
        else:
            model_name = model_config.model_name
            model_config_name = f'{model_name}ModelConfig'
            model_config_cls = MODEL_CONFIG_MAPPING[model_config_name]
            logging.info(f'creating model config from local using {model_config_name}')
            curr_model_config = model_config_cls(**model_config)
            # support definition of checkpoint path in model config
            if not curr_model_config.checkpoint_path:
                # if model_config.checkpoint_path is not provided, use checkpoint_path defined in resume
                curr_model_config.checkpoint_path = checkpoint_config.checkpoint_path

        # get common data config
        seq_len = data_config.seq_len
        forecast_horizon = data_config.forecast_horizon
        decoder_label_len = data_config.decoder_label_len
        features = data_config.features
        if features == 'M':
            input_dim = 5
            output_dim = 5
            enc_in = 5
            c_out = 5
        elif features == 'S':
            input_dim = 1
            output_dim = 1
            enc_in = 1
            c_out = 1
        else:
            raise ValueError(f'features {features} not recognized, only M, S supported')

        # set data config and fetch corresponding model config class
        curr_model_config.seq_len = seq_len
        curr_model_config.forecast_horizon = forecast_horizon
        curr_model_config.decoder_label_len = decoder_label_len
        curr_model_config.model_name = model_config.model_name
        curr_model_config.input_dim = input_dim
        curr_model_config.output_dim = output_dim
        curr_model_config.enc_in = enc_in
        curr_model_config.c_out = c_out

        return curr_model_config

    @staticmethod
    def _get_data_config(config: DictConfig):
        """
        Extract and create data configuration from main config.

        Combines common data settings with dataset-specific configurations
        and training settings (batch_size, num_workers, etc.) to create
        a complete DataConfig object.

        Args:
            config (DictConfig): Main configuration dictionary containing
                'data.common', 'data.datasets', and 'training' sections.

        Returns:
            DataConfig: Data configuration object with all dataset and
                data loading parameters set.

        Raises:
            KeyError: If required configuration sections are missing.
        """
        logging.info('Loading data config ')
        common_config = config.data.common
        dataset_configs = config.data.datasets
        training_config = config.training
        data_config = DataConfig(data_dir=common_config.data_dir,
                                 seq_len=common_config.seq_len,
                                 decoder_label_len=common_config.decoder_label_len,
                                 forecast_horizon=common_config.forecast_horizon,
                                 features=common_config.features,
                                 normalize=common_config.normalize,
                                 batch_size=training_config.batch_size,
                                 pin_memory=training_config.pin_memory,
                                 num_workers=training_config.num_workers,
                                 shuffle=training_config.shuffle,
                                 dataset_configs=dataset_configs,
                                 global_test_point=common_config.global_test_point,
                                 )
        return data_config

    @staticmethod
    def _get_training_config(config: DictConfig, local_rank, rank, world_size):
        """
        Extract and create training configuration from main config.

        Collects all configuration sections except 'data' and 'model', flattens
        them into a single dictionary, and creates the appropriate TrainerConfig
        based on the 'trainer_name' field. Adds distributed training parameters.

        Args:
            config (DictConfig): Main configuration dictionary.
            local_rank (int): Local rank for distributed training.
            rank (int): Global rank for distributed training.
            world_size (int): Total number of processes.

        Returns:
            TrainerConfig: Training configuration object with all training,
                optimization, logging, and distributed parameters set.

        Raises:
            KeyError: If 'trainer_name' is missing or trainer not found in
                TRAINER_CONFIG_MAPPING.
            ValueError: If there are overlapping keys in flattened config sections.
        """
        training_config_dict = {}
        for k, v in config.items():
            if k not in ['data', 'model']:
                training_config_dict.update(v)

        trainer_name = training_config_dict['trainer_name']
        # get training config class
        trainer_config_name = f'{trainer_name}TrainerConfig'
        training_config_cls = TRAINER_CONFIG_MAPPING[trainer_config_name]
        logging.info(f'creating training config using {trainer_config_name}')
        training_config_dict['local_rank'] = local_rank
        training_config_dict['global_rank'] = rank
        training_config_dict['world_size'] = world_size

        out_training_config = training_config_cls(**training_config_dict)

        return out_training_config

