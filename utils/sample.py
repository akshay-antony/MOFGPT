from math import e
import re
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast


def sample_smiles(model,
                  device,
                  tokenizer,
                  rl_config,
                  curr_tokens):
    """
    Sample SMILES from a trained model.
    tiles the input tokens to generate multiple sequences in parallel.
    """
    input_tokens = torch.from_numpy(curr_tokens).to(device).reshape(1, -1) # (1*len(curr_tokens))
    input_tokens = torch.tile(input_tokens, 
                              (rl_config["sampling"]["batch_size"], 1)) # (batch_size*len(curr_tokens))
    with torch.no_grad():
        if rl_config["fp16"]:
            with autocast():
                generated_sequence = model.generate(inputs=input_tokens,
                                                    max_length=rl_config["sampling"]["max_seq_len"],
                                                    do_sample=rl_config["sampling"]["do_sample"],
                                                    early_stopping=rl_config["sampling"]["early_stopping"],
                                                    num_beam_groups=rl_config["sampling"]["num_beam_groups"],
                                                    temperature=rl_config["sampling"]["temperature"],
                                                    top_k=rl_config["sampling"]["top_k"],
                                                    eos_token_id=tokenizer.eos_token_id,
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    bos_token_id=tokenizer.bos_token_id,
                                                    use_cache=True,)
        else:
            generated_sequence = model.generate(inputs=input_tokens,
                                                max_length=rl_config["sampling"]["max_seq_len"],
                                                do_sample=rl_config["sampling"]["do_sample"],
                                                early_stopping=rl_config["sampling"]["early_stopping"],
                                                num_beam_groups=rl_config["sampling"]["num_beam_groups"],
                                                temperature=rl_config["sampling"]["temperature"],
                                                top_k=rl_config["sampling"]["top_k"],
                                                eos_token_id=tokenizer.eos_token_id,
                                                pad_token_id=tokenizer.pad_token_id,
                                                bos_token_id=tokenizer.bos_token_id,
                                                use_cache=True,)
    generated_sequences = []
    for seq in generated_sequence:
        generated_sequences.append(seq)
    torch.cuda.empty_cache()
    return generated_sequences

def reward_fn(generated_sequences,
              reward_config=None):
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
            sep_count = seq.count(reward_config["sep_token_id"])
            if sep_count == 1:
                sep_idx = seq.index(reward_config["sep_token_id"])
                # check if sep is last token or second last token (check for eos)
                if sep_idx == len(seq) - 1 or sep_idx == len(seq) - 2:
                    rewards.append(reward_config["basic_rules"]["medium_reward"])
                else:
                    rewards.append(reward_config["basic_rules"]["success_reward"])
            elif sep_count == 0:
                rewards.append(reward_config["basic_rules"]["failure_reward"])
    return rewards


if __name__ == "__main__":
    pass