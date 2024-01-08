import sys
from matplotlib.pyplot import box
sys.path.append("../")
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, \
                         LlamaConfig, \
                         GPT2Config, \
                         GPT2LMHeadModel



def get_model(model_name, 
              config,
              device):
    if model_name == "llama2":
        model_config = LlamaConfig(vocab_size=config["vocab_size"],
                                   hidden_size=config["hidden_size"],
                                   intermediate_size=config["intermediate_size"],
                                   num_hidden_layers=config["num_hidden_layers"],
                                   num_attention_heads=config["num_attention_heads"],
                                   num_key_value_heads=config["num_key_value_heads"],
                                   hidden_act=config["hidden_act"],
                                   max_position_embeddings=config["max_position_embeddings"],
                                   use_cache=config["use_cache"],
                                   pad_token_id=config["pad_token_id"],
                                   bos_token_id=config["bos_token_id"],
                                   eos_token_id=config["eos_token_id"],
                                   rope_theta=config["rope_theta"],)
        model = LlamaForCausalLM(model_config).to(device)
    
    elif model_name == "gpt2":
        model_config = GPT2Config(vocab_size=config["vocab_size"],
                                  n_embd=config["n_embd"],
                                  n_layer=config["n_layer"],
                                  n_head=config["n_head"],
                                  n_inner=config["n_inner"],
                                  activation_function=config["activation_function"],
                                  max_position_embeddings=config["max_position_embeddings"],
                                  use_cache=config["use_cache"],
                                  bos_token_id=config["bos_token_id"],
                                  eos_token_id=config["eos_token_id"],
                                  resid_pdrop=config["resid_pdrop"],
                                  embd_pdrop=config["embd_pdrop"],
                                  attn_pdrop=config["attn_pdrop"],
                                  layer_norm_epsilon=config["layer_norm_epsilon"],
                                  initializer_range=config["initializer_range"],)
        model = GPT2LMHeadModel(model_config).to(device)
    return model

if __name__ == "__main__":
    pass