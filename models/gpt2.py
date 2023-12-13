import torch
import yaml
import torch.nn as nn
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel

class MOFGpt2(nn.Module):
    def __init__(self, config):
        super(MOFGpt2, self).__init__()
        self.config = config
        