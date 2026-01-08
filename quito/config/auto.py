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
    Create correponding model config, data config and training config from main config
    """

    @classmethod
    def from_config(cls, config: Union[DictConfig, str], local_rank=-1, rank=-1, world_size=1, **kwargs):
        """
        Get corresponding model, training, data config from main config
        """
        if isinstance(config, str):
            config = OmegaConf.load(config)

        # get model config
        data_config = cls._get_data_config(config)
        model_config = cls._get_model_config(config)
        training_config = cls._get_training_config(config=config, rank=rank, local_rank=local_rank,
                                                   world_size=world_size)

        return data_config, model_config, training_config

    @staticmethod
    def _get_model_config(config: DictConfig) -> Union[ModelConfig, PretrainedConfig]:
        # get model config
        model_config = config.model
        data_config = config.data.common
        # check for pretrained config (huggingface PretrainConfig) or local config (ModelConfig)
        if model_config.pretrained_model_name_or_path:
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

        else:
            model_name = model_config.model_name
            model_config_name = f'{model_name}ModelConfig'
            model_config_cls = MODEL_CONFIG_MAPPING[model_config_name]
            logging.info(f'creating model config from local using {model_config_name}')
            curr_model_config = model_config_cls(**model_config)

        # get common data config
        seq_len = data_config.seq_len
        forecast_horizon = data_config.forecast_horizon
        decoder_label_len = data_config.decoder_label_len
        # set data config and fetch corresponding model config class
        curr_model_config.seq_len = seq_len
        curr_model_config.forecast_horizon = forecast_horizon
        curr_model_config.decoder_label_len = decoder_label_len
        curr_model_config.model_name = model_config.model_name

        return curr_model_config

    @staticmethod
    def _get_data_config(config: DictConfig):
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
        Get corresponding training config from main config, will fetch all configs except data, model, then flatten them.
        Make sure there is no overlap in config keys.
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

