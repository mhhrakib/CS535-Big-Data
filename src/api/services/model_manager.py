# src/api/services/model_manager.py

import os
import glob
import threading
from functools import lru_cache
from pathlib import Path
from typing import List

import torch
import yaml

from src.main import load_config
from src.model import load_model_and_tokenizer
from src.utils import generate_summary, DOC_SEPARATOR


class ModelManager:
    """
    Singleton service to load and cache models & tokenizers based on YAML configs.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config_dir: str = "configs"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._init(config_dir)
        return cls._instance

    def _init(self, config_dir: str):
        self.config_dir = config_dir
        self.configs = {}
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Discover all YAMLs in configs/
        for cfg_file in glob.glob(os.path.join(self.config_dir, "*.yaml")):
            name = Path(cfg_file).stem
            cfg = load_config(cfg_file)
            self.configs[name] = cfg

    def get_available_models(self) -> List[str]:
        return list(self.configs.keys())

    def _load(self, name: str):
        if name not in self.configs:
            raise KeyError(f"Unknown model '{name}'")
        cfg = self.configs[name]
        model, tokenizer = load_model_and_tokenizer(
            cfg.model.name,
            device=self.device,
            ddp=False,
            local_rank=0
        )
        self.models[name] = model
        self.tokenizers[name] = tokenizer

    def get(self, name: str):
        if name not in self.models:
            self._load(name)
        return self.models[name], self.tokenizers[name]

    # def summarize(self, name: str, docs: List[str]) -> List[str]:
    #     model, tokenizer = self.get(name)
    #     cfg = self.configs[name]
    #     joined = f" {DOC_SEPARATOR} ".join(docs)
    #     # your generate_summary expects a single text, so wrap joined in a list
    #     return [generate_summary(model, tokenizer, joined, cfg, self.device)]

    def summarize(self, name: str, docs: List[str]) -> List[str]:
        """
        Summarize each document in `docs` separately.
        Returns a list of summaries, one per doc.
        """
        model, tokenizer = self.get(name)
        cfg = self.configs[name]
        summaries = []
        for text in docs:
            summ = generate_summary(model, tokenizer, text, cfg, self.device)
            summaries.append(summ)
        return summaries


# make sure this helper is visible at module top level:
@lru_cache(maxsize=1)
def get_model_manager() -> ModelManager:
    """
    Returns the global ModelManager singleton.
    """
    return ModelManager()
