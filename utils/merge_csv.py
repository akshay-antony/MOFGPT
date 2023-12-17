import glob
import os 
import numpy as np
import csv
import random
import sys
sys.path.append("../")
from dataset.dataset_gpt_1 import MOF_ID_Dataset
from tokenizer.mof_tokenizer import MOFTokenizer


def main():
    csv_folder_paths = ['../benchmark_datasets/QMOF/mofid/', 
                        '../benchmark_datasets/hMOF/mofid/', 
                        '../benchmark_datasets/Boyd&Woo/mofid/']
    csv_folder_paths = ['../benchmark_datasets/QMOF/mofid/']
    csv_file_paths = []
    for csv_folder_path in csv_folder_paths:
        csv_file_paths.extend(glob.glob(csv_folder_path + '*.csv'))
    print(csv_file_paths)

    # all_csv = pd.DataFrame()
    all_data_list = []
    all_csv = np.inf
    for csv_file_path in csv_file_paths:
        with open(csv_file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                all_data_list.append(row) 
            # appending to the list
            # all_data_list.append(curr_csv)
            # print(f"shape of {csv_file_path}: {len(curr_csv)}")
            print(f"shape of all_csv: {len(all_data_list)}")
            # curr_csv = np.array(curr_csv)

            # if isinstance(all_csv, np.ndarray):
            #     all_csv = np.concatenate((all_csv, curr_csv), axis=0)
            # else:
            #     all_csv = curr_csv

            # print(f"shape of {csv_file_path}: {curr_csv.shape}")
            # print(f"shape of all_csv: {all_csv.shape}")

    # splitting the data randomly
    # all_csv = np.array(all_data_list)
    random.shuffle(all_data_list)
    split_idx = int(len(all_data_list)*0.8)
    train_data_list = all_data_list[:split_idx]
    test_data_list = all_data_list[split_idx:]
    print(f"shape of train_data_list: {len(train_data_list)}")
    print(f"shape of test_data_list: {len(test_data_list)}")

    print(f"total number of mofs: {len(all_data_list)/10000000} million")
    vocab_path = '../tokenizer/vocab_full.txt'
    tokenizer = MOFTokenizer(vocab_path, model_max_length = 512, padding_side='right')
    test_dataset_np = np.array(test_data_list)
    torch_test_dataset = MOF_ID_Dataset(test_dataset_np, tokenizer)

    train_dataset_np = np.array(train_data_list)
    torch_train_dataset = MOF_ID_Dataset(train_dataset_np, tokenizer)
    

if __name__ == "__main__":
    main()