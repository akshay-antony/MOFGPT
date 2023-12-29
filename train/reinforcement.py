from cProfile import label
import re
import sys
sys.path.append("../")
from utils.sample import reward_fn, sample_smiles
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm


def reinforce(model,
              tokenizer,
              device,
              rl_config,
              optimizer,
              scheduler,
              save_dir,
              config):
    """
    Reinforcement learning.
    """
    # Set model to train mode.
    model.train()

    # assign sepator token id
    rl_config["reward"]["sep_token_id"] = tokenizer.convert_token_to_id(rl_config["sep_token"])
    for epoch in range(rl_config["training"]["epochs"]):
        num_steps = rl_config["num_samples"] // rl_config["sampling"]["batch_size"]
        loop = tqdm(range(num_steps),
                    leave=True,
                    colour="green",)
        for step in loop:
            curr_tokens = np.array([tokenizer.bos_token_id])
            batch_sequence = sample_smiles(model=model,
                                           device=device,
                                           tokenizer=tokenizer,
                                           rl_config=rl_config,
                                           curr_tokens=curr_tokens)

            batch_reward = reward_fn(generated_sequences=generated_sequences,
                                     reward_config=rl_config["reward"])
            loss = 0
            for seq, reward in zip(batch_sequence, batch_reward):
                eos_token_idx = seq.index(tokenizer.eos_token_id)
                seq = seq[:eos_token_idx+1]
                discounted_reward = (torch.pow(rl_config["reward"]["discount_factor"], 
                                               torch.arange(len(seq)-1, 0, -1)) * reward).to(device)
                if rl_config["training"]["fp16"]:
                    with autocast():
                        output = model(torch.tensor(seq[:-1]).unsqueeze(0).to(device),
                                       attention_mask=torch.tensor([1]*len(seq[:-1])).unsqueeze(0).to(device),
                                       labels=torch.tensor(seq[1:]).unsqueeze(0).to(device),
                                       use_cache=False,
                                       return_dict=True)
                        logits = output.logits
                else:
                    output = model(torch.tensor(seq[:-1]).unsqueeze(0).to(device),
                                   attention_mask=torch.tensor([1]*len(seq[:-1])).unsqueeze(0).to(device),
                                   labels=torch.tensor(seq[1:]).unsqueeze(0).to(device),
                                   use_cache=False,
                                   return_dict=True)
                    logits = output.logits

                log_preds = nn.functional.log_softmax(logits, dim=1)
                idxs = torch.tensor(seq[1:]).unsqueeze(0).to(device)
                action_vals = torch.gather(log_preds, 1, idxs).view(-1, 1)
                expected_reward = - torch.sum(action_vals * discounted_reward.view(-1, 1))

            loss += expected_reward

            # if rl_config["training"]["fp16"]:
            #     optimizer.backward(loss)
            # else:
            #     loss.backward()
            

            # optimizer.zero_grad()
            # loss.backward()