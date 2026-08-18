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

logging.basicConfig(level=logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('trl').setLevel(logging.ERROR)
logging.getLogger('datasets').setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

