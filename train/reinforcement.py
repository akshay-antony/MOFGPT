
import re
import sys
sys.path.append("../")
from matplotlib import use

from tokenizer.mof_tokenizer_gpt import MOFTokenizerGPT
sys.path.append("../")
from utils.sample import sample_smiles
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
import argparse
import yaml
from transformers import LlamaConfig, LlamaForCausalLM

def reward_fn(generated_sequences,
              reward_config: dict):
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
            
            # check for eos
            
    return rewards

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
        # for step in loop:
            # curr_tokens = np.array([tokenizer.bos_token_id])
            # batch_sequence = sample_smiles(model=model,
            #                                device=device,
            #                                tokenizer=tokenizer,
            #                                rl_config=rl_config,
            #                                curr_tokens=curr_tokens)

            # batch_reward = reward_fn(generated_sequences=generated_sequences,
            #                          reward_config=rl_config["reward"])
            # print(batch_reward)
            # loss = 0
            # for seq, reward in zip(batch_sequence, batch_reward):
            #     eos_token_idx = seq.index(tokenizer.eos_token_id)
            #     seq = seq[:eos_token_idx+1]
            #     discounted_reward = (torch.pow(rl_config["reward"]["discount_factor"], 
            #                                    torch.arange(len(seq)-1, 0, -1)) * reward).to(device)
            #     if rl_config["training"]["fp16"]:
            #         with autocast():
            #             output = model(torch.tensor(seq[:-1]).unsqueeze(0).to(device),
            #                            attention_mask=torch.tensor([1]*len(seq[:-1])).unsqueeze(0).to(device),
            #                            labels=torch.tensor(seq[1:]).unsqueeze(0).to(device),
            #                            use_cache=False,
            #                            return_dict=True)
            #             logits = output.logits
            #     else:
            #         output = model(torch.tensor(seq[:-1]).unsqueeze(0).to(device),
            #                        attention_mask=torch.tensor([1]*len(seq[:-1])).unsqueeze(0).to(device),
            #                        labels=torch.tensor(seq[1:]).unsqueeze(0).to(device),
            #                        use_cache=False,
            #                        return_dict=True)
            #         logits = output.logits

            #     log_preds = nn.functional.log_softmax(logits, dim=1)
            #     idxs = torch.tensor(seq[1:]).unsqueeze(0).to(device)
            #     action_vals = torch.gather(log_preds, 1, idxs).view(-1, 1)
            #     expected_reward = - torch.sum(action_vals * discounted_reward.view(-1, 1))

            # loss += expected_reward

            # if rl_config["training"]["fp16"]:
            #     optimizer.backward(loss)
            # else:
            #     loss.backward()
            

            # optimizer.zero_grad()
            # loss.backward()
    return

def make_batch_with_padding(input_token_ids,
                            device,
                            tokenizer,
                            ignore_index):
    """
    Make batch of tokens.
    """
    max_len = max([len(i) for i in input_token_ids])
    padded_token_ids = torch.zeros((0, max_len), 
                                   dtype=torch.long, 
                                   device=device)
    padded_attention_mask = torch.zeros((0, max_len),
                                        dtype=torch.long,
                                        device=device)
    padded_labels = torch.zeros((0, max_len),
                                dtype=torch.long,
                                device=device)
    
    for input_token_id in input_token_ids:
        curr_padded_token_id = torch.cat((input_token_id.reshape(-1),
                                          tokenizer.pad_token_id * torch.ones(max_len - len(input_token_id),
                                                                              dtype=torch.long,
                                                                              device=device))).reshape(1, -1)
        curr_padded_attention_mask = torch.cat((torch.ones(len(input_token_id), 
                                                          dtype=torch.long, 
                                                          device=device).reshape(-1),
                                                torch.zeros(max_len - len(input_token_id),
                                                            dtype=torch.long,
                                                            device=device).reshape(-1))).reshape(1, -1)  
        # print(f"max_len: {max_len}, len(input_token_id): {len(input_token_id)}")
        # print(curr_padded_attention_mask)
        curr_padded_labels = torch.cat((input_token_id[1:].reshape(-1),
                                        ignore_index * torch.ones(max_len - len(input_token_id) + 1,
                                                                  dtype=torch.long,
                                                                  device=device).reshape(-1))).reshape(1, -1)
        padded_token_ids = torch.cat((padded_token_ids, curr_padded_token_id), dim=0)
        padded_attention_mask = torch.cat((padded_attention_mask, curr_padded_attention_mask), dim=0)
        padded_labels = torch.cat((padded_labels, curr_padded_labels), dim=0)                                               
    return padded_token_ids, padded_attention_mask, padded_labels                 

def train_one_epoch(model,
                    tokenizer,
                    training_config,
                    sampling_config,
                    reward_config,
                    device,
                    ignore_index):
    """
    Train one epoch.
    """
    num_samplings_required = training_config["num_samples_per_epoch"] // sampling_config["batch_size"]
    loop = tqdm(range(num_samplings_required),
                leave=True,
                colour="green",)
    generated_sequences = []
    model.eval()
    for step in loop:
        curr_tokens = torch.tensor([tokenizer.bos_token_id], 
                                    device=device).unsqueeze(0)
        curr_tokens = torch.tile(curr_tokens, 
                                (sampling_config["batch_size"], 1))
        generated_sequence = sample_smiles(model=model,
                                            tokenizer=tokenizer,
                                            sampling_config=sampling_config,
                                            curr_tokens=curr_tokens)
        for seq in generated_sequence:
            generated_sequences.append(seq)
        print(len(generated_sequences))
        print(f"types: {generated_sequences[0].dtype}")
        torch.cuda.empty_cache()
        break
    
    model.train()
    nun_train_steps = training_config["num_samples_per_epoch"] // training_config["batch_size"]
    loop = tqdm(range(nun_train_steps),
                leave=True,
                colour="green",)
    for step in loop:
        # sample from generated sequences
        sampled_indices = np.random.choice(len(generated_sequences),
                                           training_config["batch_size"],
                                           replace=False)
        sampled_sequences = [generated_sequences[i] for i in sampled_indices]
        for seq in sampled_sequences:
            print(len(seq))
        # make batch
        padded_token_ids, \
        padded_attention_mask, \
        padded_labels = make_batch_with_padding(input_token_ids=sampled_sequences,
                                                device=device,
                                                tokenizer=tokenizer,
                                                ignore_index=ignore_index)
        # print(padded_token_ids.shape)
        # print(padded_attention_mask.shape)
        # print(padded_labels.shape)
        # print("token_ids")
        # print(padded_token_ids[0])
        # print("attention_mask")
        # print(padded_attention_mask[0])
        # print("labels")
        # print(padded_labels[0])
        # print("########## 1 ##########")
        # print("token_ids")
        # print(padded_token_ids[1])
        # print("attention_mask")
        # print(padded_attention_mask[1])
        # print("labels")
        # print(padded_labels[1])
        if training_config["fp16"]:
            with autocast():
                output = model(input_ids=padded_token_ids,
                               attention_mask=padded_attention_mask,
                               labels=padded_labels,
                               use_cache=False,
                               
        print(padded_token_ids.shape)
        return
        # loop.set_description(f"Loss: {loss.item():.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename",
                        type=str,
                        default="../config/config.yaml",
                        help="Main config filename")
    args = parser.parse_args()
    config = yaml.load(open(args.config_filename, "r"), Loader=yaml.FullLoader)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # separate config
    rl_config = config["reinforcement_learning"]
    tokenizer_config = config["data"]["tokenizer"]
    model_config = config["model"]
    data_config = config["data"]
    
    tokenizer = MOFTokenizerGPT(vocab_file=data_config["vocab_path"],
                                add_special_tokens=tokenizer_config["add_special_tokens"],  
                                truncation=tokenizer_config["truncation"],
                                pad_token=tokenizer_config["pad_token"],
                                bos_token=tokenizer_config["bos_token"],
                                eos_token=tokenizer_config["eos_token"],
                                mask_token=tokenizer_config["mask_token"],
                                unk_token=tokenizer_config["unk_token"],
                                max_len=tokenizer_config["max_seq_len"],)
    model_config["vocab_size"] = tokenizer.vocab_size
    model_config["pad_token_id"] = tokenizer.pad_token_id
    model_config["bos_token_id"] = tokenizer.bos_token_id
    model_config["eos_token_id"] = tokenizer.eos_token_id
    model_config["max_position_embeddings"] = tokenizer_config["max_seq_len"]
    reward_config = rl_config["reward"]
    reward_config["sep_token_id"] = tokenizer.convert_token_to_id(reward_config["sep_token"])

    # Load model
    llama_model_config = LlamaConfig(vocab_size=model_config["vocab_size"],
                                     hidden_size=model_config["hidden_size"],
                                     intermediate_size=model_config["intermediate_size"],
                                     num_hidden_layers=model_config["num_hidden_layers"],
                                     num_attention_heads=model_config["num_attention_heads"],
                                     num_key_value_heads=model_config["num_key_value_heads"],
                                     hidden_act=model_config["hidden_act"],
                                     max_position_embeddings=model_config["max_position_embeddings"],
                                     use_cache=model_config["use_cache"],
                                     pad_token_id=model_config["pad_token_id"],
                                     bos_token_id=model_config["bos_token_id"],
                                     eos_token_id=model_config["eos_token_id"],
                                     rope_theta=model_config["rope_theta"],)
    model = LlamaForCausalLM(llama_model_config).to(device)
    print("Model loaded with {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # load optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=rl_config["training"]["optimizer"]["lr"],
                                  weight_decay=rl_config["training"]["optimizer"]["weight_decay"])

    # loading state dict
    saved_dict = torch.load(rl_config["training"]["saved_state_dict_filename"])
    if model.load_state_dict(saved_dict["model_state_dict"]):
        print("Model state dict loaded")

    for epoch in range(rl_config["training"]["epochs"]):
        # train one epoch
        train_one_epoch(model=model,
                        tokenizer=tokenizer,
                        training_config=rl_config["training"],
                        sampling_config=rl_config["sampling"],
                        reward_config=rl_config["reward"],
                        device=device,
                        ignore_index=config["data"]["ignore_index"])
        return    

if __name__ == "__main__":
    main()