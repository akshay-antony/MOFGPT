from calendar import c
from math import e
from os import sep
import re
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast


def sample_smiles(model,
                  tokenizer,
                  sampling_config,
                  curr_tokens):
    """
    Sample SMILES from a trained model.
    tiles the input tokens to generate multiple sequences in parallel.
    """
    with torch.no_grad():
        generated_sequence = model.generate(inputs=curr_tokens,
                                            max_length=sampling_config["max_seq_len"],
                                            do_sample=sampling_config["do_sample"],
                                            early_stopping=sampling_config["early_stopping"],
                                            num_beam_groups=sampling_config["num_beam_groups"],
                                            temperature=sampling_config["temperature"],
                                            top_k=sampling_config["top_k"],
                                            eos_token_id=tokenizer.eos_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            bos_token_id=tokenizer.bos_token_id,
                                            use_cache=True,)
     
    generated_sequences = []
    for seq in generated_sequence:
        generated_sequences.append(seq)
        # print(len(seq))
        eos_token_idx = (seq == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        # print(eos_token_idx)
        generated_sequences.append(seq[:eos_token_idx[0]+1] if len(eos_token_idx) > 0 else seq)
    torch.cuda.empty_cache()
    return generated_sequences

def reward_fn(generated_sequences,
              reward_config,
              eos_token_id,):
    """
    Reward function for RL.
    args:
        generated_sequences: list of generated sequences, List[int]
        name: name of the reward function, str
        reward_config: reward configuration, dict
    return:
        rewards: list of rewards, List[float]
    """
    rewards = []
    if reward_config["name"] == "basic_rules":
        for seq in generated_sequences:
            curr_reward = 0

            if seq[-1] == eos_token_id:
                curr_reward += reward_config["basic_rules"]["eos_reward"]
            else:
                curr_reward += reward_config["basic_rules"]["failure_reward"]

            sep_count = seq.count(reward_config["sep_token_id"])
            if sep_count == 1:
                curr_reward += reward_config["basic_rules"]["single_sep_reward"]
            elif sep_count > 1:
                curr_reward += reward_config["basic_rules"]["multiple_sep_reward"]
            elif sep_count == 0:
                curr_reward += reward_config["basic_rules"]["no_sep_reward"]
            rewards.append(curr_reward)
    return rewards


if __name__ == "__main__":
    pass