import re
import sys
sys.path.append("../")
import torch 
import numpy as np
from tqdm import tqdm
import yaml
from tokenizer.mof_tokenizer_gpt import MOFTokenizerGPT
import glob
from split_csv import split_csv
import os
import csv
import json


def count_unique_tokens(data, 
                        tokenizer):
    max_unique_tokens = tokenizer.vocab_size
    unique_tokens = np.zeros(max_unique_tokens)
    for str_no, curr_str in enumerate(tqdm(data,
                                           desc='Counting unique tokens',
                                           colour='green',
                                           total=len(data))):
        token_ids = np.asarray(tokenizer.encode(curr_str[0]))
        unique_tokens[token_ids] += 1
        # if str_no % 1000 == 0 and str_no != 0:
        #     break
    unique_tokens_str = {}
    for i in range(len(unique_tokens)):
        unique_tokens_str[tokenizer.convert_id_to_token(i)] = unique_tokens[i]
    # sort by value
    unique_tokens_str = dict(sorted(unique_tokens_str.items(), key=lambda item: item[1], reverse=True))
    for i in unique_tokens_str:
        if unique_tokens_str[i] != 0:
            print(f"{i}: {unique_tokens_str[i]}")
    save_filename = "./unique_tokens.json"
    with open(save_filename, "w") as f:
        json.dump(unique_tokens_str, f)

def visualize_unique_tokens(unique_tokens_str):
    # import matplotlib
    # matplotlib.use('TkAgg')  # Or another backend that suits your environment
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(20, 10))
    plt.bar(unique_tokens_str.keys(), unique_tokens_str.values())
    plt.xticks(rotation=90)
    # plt.show()
    plt.savefig("./unique_tokens.png")

def main():
    save_filename = "./unique_tokens.json"
    with open(save_filename, "r") as f:
        unique_tokens_str = json.load(f)
    
    # visualize_unique_tokens(unique_tokens_str)
    non_zero_tokens_str = {}
    num_zero_count_tokens = 0
    non_zero_count_tokens = 0
    for key in unique_tokens_str:
        if unique_tokens_str[key] == 0:
            print(key)
            num_zero_count_tokens += 1
        else:
            non_zero_count_tokens += 1
            non_zero_tokens_str[key] = unique_tokens_str[key]

    print(f"num_zero_count_tokens: {num_zero_count_tokens}")
    print(f"non_zero_count_tokens: {non_zero_count_tokens}")
    visualize_unique_tokens(non_zero_tokens_str)
    return

def main2():
    config_filename = "../config.yaml"
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)
    config_tokenizer = config["data"]["tokenizer"]
    tokenizer = MOFTokenizerGPT(vocab_file="../tokenizer/vocab_with_eos.txt",
                                add_special_tokens=config_tokenizer["add_special_tokens"],
                                truncation=config_tokenizer["truncation"],
                                pad_token=config_tokenizer["pad_token"],
                                mask_token=config_tokenizer["mask_token"],
                                bos_token=config_tokenizer["bos_token"],
                                eos_token=config_tokenizer["eos_token"],
                                unk_token=config_tokenizer["unk_token"],
                                max_len=config_tokenizer["max_seq_len"],)
    csv_filenames = []
    for csv_folder_path in config["data"]["csv_folder_paths"]:
        csv_folder_path = os.path.join("../", csv_folder_path)
        csv_filenames.extend(glob.glob(os.path.join(csv_folder_path, "*.csv")))
    
    all_data_list = []
    for csv_filename in csv_filenames:
        print(f"Counting unique tokens in {csv_filename}")
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                all_data_list.append(row)
    print(f"shape of all_csv: {len(all_data_list)}")    
    # count_unique_tokens(all_data_list, tokenizer)

    
if __name__ == "__main__":
    main()