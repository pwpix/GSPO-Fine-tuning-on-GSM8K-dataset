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


# Reward model
# For mathematical resaon we need to:
#       1. Extract the numerical answer from free-form text
#       2. Check correctness by comparing to the ground truth answer
#       3. Award partial credit for reasonable attempts even if the final answer is wrong


# Implementation of GSM8K Reward Signal
class GSM8KRewardSignal:
    """Reward model for GSM8K mathematical reasoning in GSPO"""

    def extract_numerical_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from text"""

        #Strategy 1: Look for #### (GSM8K format)
        if "####" in text:
            answer = text.split("####")[-1].strip()
            answer = answer.replace(',', '').replace('$', '')
            try:
                return float(answer)
            except:
                return None
        
        #Strategy 2: Regex patterns
        patterns = [
            r"(?:The answer is|answer:|Answer:)\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?",
            r"(?:equals?|=)\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
            r"(?:total|sum|result)\s*(?:is|:|=)?\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1].replace(',', ''))
                except:
                    continue

        #Strategy 3: Any number
        numbers = re.findall(r'([+-]?\d+(?:,\d{3}*(?:\.\d+)?)', text)
        if numbers:
            try:
                return float(numbers[-1].replace(',', ''))
            except:
                pass
        return None

    def compute_reward(self, response: str, correct_answer: float, question: str = None) -> float:
        """Compute reward for a response"""

        predicted = self.extract_numerical_answer(response)
        has_calculation = any(word in response.lower() 
                          for word in ['=', '+', '-', '*', '/', 'multiply', 'divide', 'add', 'subtract'])
        has_steps = len(response.split('.')) > 2
        has_numbers = bool(re.search(r'\d', response))

        #No parsable answer
        if predicted is None:
            if len(response) > 200 and has_calculation and has_numbers:
                return 0.2
            elif has_calculation and has_numbers:
                return 0.1
            else:
                return 0.05

        #Correct answer
        if abs(predicted - correct_answer) < 0.01:
            return 1.0 + (0,2 if has_steps else 0)
    
        #Wrong answer - partial credit
        relative_error = abs(predicted - correct_answer) / (abs(correct_answer) + 1e-10)

        if relative_error <0.1:
            reward = 0.8
        elif relative_error < 0.3:
            reward = 0.5
        else:
            reward = 0.15

        if has_calculation and has_steps:
            reward += 0.1

            return reward

reward_model = GSM8KRewardSignal()
print("Reward model initialized")



























