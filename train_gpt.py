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
from transformers import GPT2Config, GPT2LMHeadModel


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--config_filename", 
                      default="config.yaml", 
                      type=str,
                      help="Path to config file")
    args = args.parse_args()

    with open(args.config_filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # load tokenizer
    # tokenizer = MOFTokenizer(config['data']['vocab_path'], 
    #                          model_max_length = config['data']['max_seq_len'], 
    #                          padding_side='right')
    tokenizer = MOFTokenizerGPT(config['data']['vocab_path'],
                                )
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
    train_dataset = MOF_ID_Dataset(train_data_np, tokenizer)
    print("For test dataset:")
    test_dataset = MOF_ID_Dataset(test_data_np, tokenizer)
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
        token_ids, labels = batch
        print(f"token_ids shape: {token_ids.shape}")
        print(token_ids[0])
        if b_no >= 100:
            break
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