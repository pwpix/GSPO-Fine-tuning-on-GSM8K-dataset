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
