import torch
import yaml
import os
import glob
import sys
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataset.dataset_finetune_transformer import MOF_ID_Dataset
from tokenizer.mof_tokenizer import MOFTokenizer
from tokenizer.mof_tokenizer_gpt import MOFTokenizerGPT
from utils.split_csv import split_csv
from transformers import GPT2Config, \
                         GPT2LMHeadModel, \
                         LlamaConfig, \
                         LlamaForCausalLM
from transformers import get_constant_schedule_with_warmup


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--config_filename", 
                      default="config.yaml", 
                      type=str,
                      help="Path to config file")
    args = args.parse_args()

    with open(args.config_filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config_data = config['data']
    config_tokenizer = config_data['tokenizer']
    config_model = config['model']
    config_training = config['training']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = MOFTokenizerGPT(vocab_file=config['data']['vocab_path'],
                                add_special_tokens=config_tokenizer['add_special_tokens'],
                                truncation=config_tokenizer['truncation'],
                                pad_token=config_tokenizer['pad_token'],
                                mask_token=config_tokenizer['mask_token'],
                                bos_token=config_tokenizer['bos_token'],
                                eos_token=config_tokenizer['eos_token'],
                                unk_token=config_tokenizer['unk_token'],
                                max_len=config_tokenizer['max_seq_len'],)

    config_model['vocab_size'] = tokenizer.vocab_size
    print(f"tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"tokenizer vocab dict: {tokenizer.get_vocab()}")


    # return
    csv_filenames = []
    for csv_folder_path in config['data']['csv_folder_paths']:
        csv_filenames.extend(glob.glob(csv_folder_path + '*.csv'))
    train_data_np, \
    test_data_np = split_csv(csv_filenames,
                             train_test_ratio = config['data']['train_test_ratio'],
                             random_seed = config['random_seed'])

    print("For train dataset:")
    train_dataset = MOF_ID_Dataset(train_data_np, 
                                   tokenizer,
                                   config_data['ignore_index'])
    print("For test dataset:")
    test_dataset = MOF_ID_Dataset(test_data_np, 
                                  tokenizer,
                                  config_data['ignore_index'])
    # return
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config['data']['batch_size'], 
                                  shuffle=True, 
                                  num_workers=config['data']['num_workers'],
                                  collate_fn=train_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config['data']['batch_size'], 
                                 shuffle=True, 
                                 num_workers=config['data']['num_workers'],
                                 collate_fn=test_dataset.collate_fn)
    # testing batch loading
    for b_no, batch in enumerate(train_dataloader):
        token_ids = batch['token_ids']
        mask_ids = batch['mask_ids']
        target_token_ids = batch['target_token_ids']
        label = batch['label']
        print(f"token_ids shape: {token_ids.shape}, mask_ids shape: {mask_ids.shape}, label shape: {label.shape}")
        print(token_ids[0])
        print(mask_ids[0])
        eos_token_pos = torch.where(token_ids[0] == tokenizer.eos_token_id)[0]
        num_non_masked_tokens = torch.sum(mask_ids[0])
        print(f"eos token pos: {eos_token_pos}")
        print(f"num_non_masked_tokens: {num_non_masked_tokens}")
        print(f"target_token_ids: {target_token_ids[0]}")
        if b_no >= 2:
            break

    # Model config
    # adding special token ids to model config
    config_model['pad_token_id'] = tokenizer.pad_token_id
    config_model['bos_token_id'] = tokenizer.bos_token_id
    config_model['eos_token_id'] = tokenizer.eos_token_id
    config_model['vocab_size'] = tokenizer.vocab_size
    config_model['max_position_embeddings'] = config_tokenizer['max_seq_len']

    model_config = LlamaConfig(vocab_size=config_model['vocab_size'],
                               hidden_size=config_model['hidden_size'],
                               intermediate_size=config_model['intermediate_size'],
                               num_hidden_layers=config_model['num_hidden_layers'],
                               num_attention_heads=config_model['num_attention_heads'],
                               num_key_value_heads=config_model['num_key_value_heads'],
                               hidden_act=config_model['hidden_act'],
                               max_position_embeddings=config_model['max_position_embeddings'],
                               use_cache=config_model['use_cache'],
                               pad_token_id=config_model['pad_token_id'],
                               bos_token_id=config_model['bos_token_id'],
                               eos_token_id=config_model['eos_token_id'],
                               rope_theta=config_model['rope_theta'],)
    model = LlamaForCausalLM(model_config).to(device)
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6}M")

    # optimizer
    optimizer = getattr(torch.optim,
                        config_training['optimizer']['type'])
    optimizer = optimizer(model.parameters(), 
                          lr=config_training['optimizer']['lr'],
                          weight_decay=config_training['optimizer']['weight_decay'])
    # scheduler
    if config_training['scheduler']['type'] == 'cosine':
        num_training_steps = (len(train_dataloader) / config_training['gradient_accumulation_steps']) * config_training['num_epochs']
        num_warmup_steps = int(num_training_steps * config_training['scheduler']['warmup_ratio'])
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=num_warmup_steps,
                                                      num_training_steps=num_training_steps)
        print(f"created cosine scheduler with num_training_steps: {num_training_steps}, num_warmup_steps: {num_warmup_steps}")
    # training
    for epoch in range(config_training['num_epochs']):
        loop = tqdm(train_dataloader,
                    desc=f"Epoch {epoch}",
                    colour='green',
                    total=len(train_dataloader))
        for b_no, batch in enumerate(loop):
            token_ids = batch['token_ids'].to(device)
            mask_ids = batch['mask_ids'].to(device)
            target_token_ids = batch['target_token_ids'].to(device)
            label = batch['label'].to(device)
            outputs = model(input_ids=token_ids,
                            attention_mask=mask_ids,
                            labels=target_token_ids,
                            use_cache=config_model['use_cache'])
    return

    # setting model config
    print(f"Voab size: {tokenizer.vocab_size}")
    print(f"tokens: {list(tokenizer.get_vocab().keys())}")
    model_config = GPT2Config(n_positions=config['data']['max_seq_len'])
    model, loading_info = GPT2LMHeadModel.from_pretrained(config['model']['pretrained_model_name'],
                                                          config=model_config,
                                                          ignore_mismatched_sizes=True,
                                                          output_loading_info=True)
    print(f"Missing keys: {loading_info['missing_keys']}")
    print(f"Unexpected keys: {loading_info['unexpected_keys']}")
    print(f"Error keys: {loading_info}")
    print(f"eos token id: {tokenizer.eos_token_id}")
 
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

if __name__ == "__main__":
    main()