import os
import yaml
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class DataConfig:
    dataset_name: str
    sample_ratio: float
    max_input_length: int
    max_output_length: int
    batch_size: int
    num_workers: int
    seed: int


@dataclass
class ModelConfig:
    model_name: str
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float
    gradient_accumulation_steps: int


@dataclass
class TrainingConfig:
    output_dir: str
    num_epochs: int
    eval_steps: int
    save_steps: int
    logging_steps: int
    use_fp16: bool
    distributed: bool
    local_rank: int
    world_size: int


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        data_config = DataConfig(**config_dict['data'])
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])

        return cls(data=data_config, model=model_config, training=training_config)