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
import wandb

def reward_fn(generated_sequences,
              reward_config,
              eos_token_id,):
    """
    Reward function for RL.
    args:
        generated_sequences: list of generated sequences, List[tensor[int]]
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
                curr_reward += reward_config["basic_rules"]["no_eos_reward"]

            sep_count = (seq == reward_config["sep_token_id"]).sum().item()
            if sep_count == 1:
                curr_reward += reward_config["basic_rules"]["single_sep_reward"]
            elif sep_count > 1:
                curr_reward += reward_config["basic_rules"]["multiple_sep_reward"]
            elif sep_count == 0:
                curr_reward += reward_config["basic_rules"]["no_sep_reward"]
            rewards.append(curr_reward)
    return rewards

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
    sequence_lens = []
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
        sequence_lens.append(len(input_token_id)-1)
        padded_token_ids = torch.cat((padded_token_ids, curr_padded_token_id), dim=0)
        padded_attention_mask = torch.cat((padded_attention_mask, curr_padded_attention_mask), dim=0)
        padded_labels = torch.cat((padded_labels, curr_padded_labels), dim=0)                                               
    return padded_token_ids, padded_attention_mask, padded_labels, sequence_lens                 

def train_one_epoch(model,
                    tokenizer,
                    training_config,
                    sampling_config,
                    reward_config,
                    model_config,
                    device,
                    ignore_index,
                    optimizer,
                    epoch,
                    scaler=None,):
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
        torch.cuda.empty_cache()
    
    model.train()
    nun_train_steps = training_config["num_samples_per_epoch"] // training_config["batch_size"]
    loop = tqdm(range(nun_train_steps),
                leave=True,
                colour="green",)
    
    epoch_loss = 0
    epoch_reward = 0
    epoch_element_count = 0
    for step in loop:
        # sample from generated sequences
        batch_reward = 0
        batch_loss = 0
        sampled_indices = np.random.choice(len(generated_sequences),
                                           training_config["batch_size"],
                                           replace=False)
        sampled_sequences = [generated_sequences[i] for i in sampled_indices]
        rewards = reward_fn(generated_sequences=sampled_sequences,
                            reward_config=reward_config,
                            eos_token_id=tokenizer.eos_token_id)
        rewards = torch.tensor(rewards,
                               dtype=torch.float,
                               device=device).reshape(-1, 1) # (batch_size, 1)
        
        # make batch
        padded_token_ids, \
        padded_attention_mask, \
        padded_labels, \
        seq_lens = make_batch_with_padding(input_token_ids=sampled_sequences,
                                           device=device,
                                           tokenizer=tokenizer,
                                           ignore_index=ignore_index)
        batch_count = padded_token_ids.shape[0]
        if training_config["fp16"]:
            with autocast():
                output = model(input_ids=padded_token_ids,
                               attention_mask=padded_attention_mask,
                               labels=padded_labels,
                               use_cache=model_config["use_cache"],
                               return_dict=model_config["return_dict"],
                               output_attentions=model_config["output_attentions"],
                               output_hidden_states=model_config["output_hidden_states"])
        else:
            output = model(input_ids=padded_token_ids,
                           attention_mask=padded_attention_mask,
                           labels=padded_labels,
                           use_cache=model_config["use_cache"],
                           return_dict=model_config["return_dict"],
                           output_attentions=model_config["output_attentions"],
                           output_hidden_states=model_config["output_hidden_states"])
        
        curr_loss = output.loss.item()
        curr_logits = output.logits
        log_preds = torch.nn.functional.log_softmax(curr_logits, dim=-1)
        for b_no, log_pred in enumerate(log_preds):
            discounted_returns = (torch.pow(reward_config["discount_factor"], 
                                            torch.arange(seq_lens[b_no], 
                                                            0, 
                                                            -1, 
                                                            device=device)) * rewards[b_no]).reshape(-1, 1).to(device)
            action_values = log_pred[torch.arange(seq_lens[b_no]), 
                                     padded_labels[b_no, :seq_lens[b_no]]].reshape(-1, 1)
            batch_reward += rewards[b_no]    
            expected_reward = -torch.sum(discounted_returns * action_values)
            batch_loss += expected_reward
        batch_loss /= batch_count
        batch_reward /= batch_count
        epoch_loss += batch_loss.item() * batch_count
        epoch_reward += batch_reward.item() * batch_count
        epoch_element_count += batch_count

        if training_config["fp16"]:
            scaler.scale(batch_loss).backward()
        else:
            batch_loss.backward()

        loop.set_description(f"Epoch [{epoch+1}/{training_config['epochs']}]")
        loop.set_postfix(loss=batch_loss.item(),
                         reward=batch_reward.item())
        
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    return epoch_loss / epoch_element_count, epoch_reward / epoch_element_count

def evaluate(model,
             tokenizer,
             eval_config,
             sampling_config,
             reward_config,
             model_config,
             device,
             ignore_index,):
    """
    Evaluate the model by generating SMILES.
    """
    model.eval()
    curr_tokens = torch.tensor([tokenizer.bos_token_id],
                               device=device).unsqueeze(0)
    curr_tokens = torch.tile(curr_tokens,
                             (eval_config["num_samples"], 1))
    print(f"Sampling {eval_config['num_samples']} sequences")
    generated_sequences = sample_smiles(model=model,
                                        tokenizer=tokenizer,
                                        sampling_config=sampling_config,
                                        curr_tokens=curr_tokens)
    eval_reward = 0
    for seq in generated_sequences:
        eval_reward += reward_fn(generated_sequences=[seq],
                                 reward_config=reward_config,
                                 eos_token_id=tokenizer.eos_token_id)[0]
    eval_reward /= len(generated_sequences)
    return eval_reward
    
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
    rl_model_config = rl_config["model"]
    wandb.init(project=rl_config["training"]["model_name"])
    wandb.config.update(config)
    
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

    if rl_config["training"]["fp16"]:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    eval_config = rl_config["eval"]
    best_reward = -np.inf
    for epoch in range(rl_config["training"]["epochs"]):
        print(f"Epoch {epoch+1}/{rl_config['training']['epochs']}")
        # train one epoch
        train_epoch_loss, \
        train_epoch_reward = train_one_epoch(model=model,
                                             tokenizer=tokenizer,
                                             training_config=rl_config["training"],
                                             sampling_config=rl_config["sampling"],
                                             reward_config=rl_config["reward"],
                                             model_config=rl_model_config,
                                             device=device,
                                             ignore_index=config["data"]["ignore_index"],
                                             optimizer=optimizer,
                                             scaler=scaler,
                                             epoch=epoch,)
        print(f"Train epoch loss: {train_epoch_loss}")
        print(f"Train epoch reward: {train_epoch_reward}")
        if epoch % eval_config["eval_interval"] == 0:
            eval_reward = evaluate(model=model,
                                   tokenizer=tokenizer,
                                   eval_config=eval_config,
                                   sampling_config=rl_config["sampling"],
                                   reward_config=rl_config["reward"],
                                   model_config=rl_model_config,
                                   device=device,
                                   ignore_index=config["data"]["ignore_index"],)
            print(f"Eval reward: {eval_reward}")
            if eval_reward > best_reward:
                best_reward = eval_reward
                save_filename = f"{rl_config['training']['save_dir']}/best_reward_{rl_config['training']['model_name']}.pt"
                torch.save({"model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "best_reward": best_reward,}, 
                           save_filename)
                print("Model saved to {}".format(save_filename))
        wandb.log({"train_epoch_loss": train_epoch_loss,
                   "train_epoch_reward": train_epoch_reward,
                   "eval_reward": eval_reward,
                   "epoch": epoch})

if __name__ == "__main__":
    main()