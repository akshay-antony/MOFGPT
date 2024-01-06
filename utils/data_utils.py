import glob
import numpy as np
import csv
import random
import sys
sys.path.append("../")


def split_csv(csv_filenames,
              train_test_ratio = 0.8,
              random_seed = 42):
    all_data_list = []

    for csv_filename in csv_filenames:
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                all_data_list.append(row) 
    
    print(f"shape of all_csv before duplicate removal: {len(all_data_list)}")
    all_data_list = remove_duplicates(all_data_list)
    print(f"shape of all_csv after duplicate removal: {len(all_data_list)}")
    random.seed(random_seed)
    random.shuffle(all_data_list)
    split_idx = int(len(all_data_list)*train_test_ratio)
    train_data_list = all_data_list[:split_idx]
    test_data_list = all_data_list[split_idx:]
    print(f"shape of train_data_list: {len(train_data_list)}")
    print(f"shape of test_data_list: {len(test_data_list)}")
    return np.array(train_data_list), np.array(test_data_list)

def remove_duplicates(data):
    unq_set = set()
    num_duplicates = 0
    refined_data = []

    for i in data:
        if i[0] in unq_set:
            num_duplicates += 1
        else:
            unq_set.add(i[0])
            refined_data.append(i)
    print(f"Number of duplicates: {num_duplicates}")
    assert len(refined_data) == len(unq_set)
    return refined_data

if __name__ == " __main__":
    pass