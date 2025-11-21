import logging
from omegaconf import DictConfig

from quito.config.model import ModelType, PatchTSTModelConfig, DLinearModelConfig, ChronosModelConfig, MoriaiModelConfig, HuggingFaceModelConfig
from quito.config.training import TrainingConfig, TrainerType
from quito.config.data import DataConfig

# model_name -> model_config, trainer_name -> trainer_config structure

MODEL_CONFIG_MAPPING = {
    ModelType.PATCHTST: PatchTSTModelConfig,
    ModelType.DLINEAR: DLinearModelConfig,
    ModelType.CHRONOS: ChronosModelConfig,
    ModelType.MORIAI: MoriaiModelConfig,
    ModelType.HUGGINGFACE: HuggingFaceModelConfig
}

TRAINER_CONFIG_MAPPING = {
    TrainerType.TRAINER: TrainingConfig
}


class AutoConfig:
    """
    Create correponding model config, data config and training config from main config
    """
    @classmethod
    def from_config(cls, config: DictConfig, local_rank=-1, rank=-1, world_size=1, **kwargs):
        """
        Get corresponding model, training, data config from main config
        """
        # get model config 
        data_config = cls._get_data_config(config)
        model_config = cls._get_model_config(config)
        training_config = cls._get_training_config(config=config, rank=rank, local_rank=local_rank, world_size=world_size)

        return data_config, model_config, training_config

    @staticmethod
    def _get_model_config(config: DictConfig):
        # get model config 
        model_config = config.model
        data_config = config.data.common
        # get common data config
        seq_len = data_config.seq_len
        forecast_horizon = data_config.forecast_horizon
        decoder_label_len = data_config.decoder_label_len
        # set data config and fetch corresponding model config class
        model_config.seq_len = seq_len
        model_config.forecast_horizon = forecast_horizon
        model_config.decoder_label_len = decoder_label_len
        
        model_type = ModelType(model_config.model_name)
        model_config_cls = MODEL_CONFIG_MAPPING[model_type]

        logging.info(f'creating model config using {model_type}')

        out_model_config = model_config_cls(**model_config)
        
        return out_model_config
    
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
        
        trainer_name = TrainerType(training_config_dict['trainer_name'])
        # get training config class
        training_config_cls = TRAINER_CONFIG_MAPPING[trainer_name]
        logging.info(f'creating training config using {trainer_name}')
        training_config_dict['local_rank'] = local_rank
        training_config_dict['global_rank'] = rank
        training_config_dict['world_size'] = world_size

        out_training_config = training_config_cls(**training_config_dict)

        return out_training_config
