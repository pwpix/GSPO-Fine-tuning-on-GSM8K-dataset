"""
    This tutorial demonstrates how to fine tune deepseek-math-7b-base on GSM8K dataset for improved mathematical reasoning
    using Group Sequence Policy Optimization (GSPO) on AMD GPUs
"""

import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

from trl import GRPOConfig, GRPOTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback
)

from datasets import load_dataset

import torch
from typing import Optional, List
import numpy as np
import re
import logging
from datetime import datetime
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

print("All imports successful")

logging.basicConfig(level=logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('trl').setLevel(logging.ERROR)
logging.getLogger('datasets').setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# Centralized configuration dataclass to hold all training hyperparameters

@dataclass
class GSPOTrainingConfig:
    """GSPO (Group Sequence Policy Optimization) training configuration"""
    model_name: str = field(default="deepseek-ai/deepseek-math-7b-base")
    output_dir: str = field(default="./gspo_finetuned_model")

    #training parameters
    num_train_epochs: int  = field(default = 2)
    per_device_train_batch_size: int = field(default = 2)
    gradient_accumulation_steps:int = field(default = 4)
    learning_rate:float = field(default = 1e-5)


    #GSPO-specific: Generation parameters
    num_generations: int = field(default = 4)
    temperature: float = field(default = 0.7)
    max_new_tokens: int = field(default = 256)

    #GSPO-specific: Policy Optimization parameters
    steps_per_generation:int = field(default = 16)
    epsilon: float = field(default = 3e-4)
    epsilon_high:float = field(default = 4e-4)

    #Dataset parameters
    max_prompt_length:int = field(default = 512)
    train_split_ratio:float = field(default = 0.8)

    #Evaluation parameters
    eval_steps: int = field(default = 20)
    save_steps: int = field(default = 100)
    logging_steps: int = field(default = 5)

config = GSPOTrainingConfig(
    num_train_epochs = 1,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate = 1e-5,
    num_generations = 4,
    temperature = 0.7,
    max_new_tokens = 256,
    eval_steps = 20,
    steps_per_generation = 16,
    epsilon = 2e-4,
    epsilon_high = 5e-4,
    logging_steps = 5,
)

print("GSPO configuration created")
