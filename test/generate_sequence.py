import sys
sys.path.append("../")
import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from tokenizer.mof_tokenizer_gpt import MOFTokenizerGPT
from transformers import LlamaForCausalLM, LlamaConfig
from torch.cuda.amp import autocast
import wandb
import yaml


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config_filename = "../config/config.yaml"
    generate_config_filename = "../config/config_generate.yaml"
    with open(train_config_filename, "r") as f:
        train_config = yaml.safe_load(f)
    with open(generate_config_filename, "r") as f:
        generate_config = yaml.safe_load(f)

    config_model = train_config["model"]
    config_tokenizer = train_config["data"]["tokenizer"]
    
    tokenizer = MOFTokenizerGPT(vocab_file=train_config["data"]["vocab_path"],
                                add_special_tokens=config_tokenizer["add_special_tokens"],
                                truncation=config_tokenizer["truncation"],
                                pad_token=config_tokenizer["pad_token"],
                                mask_token=config_tokenizer["mask_token"],
                                bos_token=config_tokenizer["bos_token"],
                                eos_token=config_tokenizer["eos_token"],
                                unk_token=config_tokenizer["unk_token"],
                                max_len=config_tokenizer["max_seq_len"],)
    
    config_model["vocab_size"] = tokenizer.vocab_size
    config_model["pad_token_id"] = tokenizer.pad_token_id
    config_model["bos_token_id"] = tokenizer.bos_token_id
    config_model["eos_token_id"] = tokenizer.eos_token_id
    config_model["max_position_embeddings"] = config_tokenizer["max_seq_len"]
    model_config = LlamaConfig(vocab_size=config_model["vocab_size"],
                               hidden_size=config_model["hidden_size"],
                               intermediate_size=config_model["intermediate_size"],
                               num_hidden_layers=config_model["num_hidden_layers"],
                               num_attention_heads=config_model["num_attention_heads"],
                               num_key_value_heads=config_model["num_key_value_heads"],
                               hidden_act=config_model["hidden_act"],
                               max_position_embeddings=config_model["max_position_embeddings"],
                               use_cache=config_model["use_cache"],
                               pad_token_id=config_model["pad_token_id"],
                               bos_token_id=config_model["bos_token_id"],
                               eos_token_id=config_model["eos_token_id"],
                               rope_theta=config_model["rope_theta"],)
    model = LlamaForCausalLM(model_config).to(device)

    training_dict = torch.load(generate_config["state_dict_filename"])
    if model.load_state_dict(training_dict["model_state_dict"]):
        print("Model loaded successfully.")
    else:
        print("Model loaded with errors.")

    model.eval()
    print(f"model.generate_config: {model.generation_config}") 
    # model.generate()

    input_smiles = "[BOS]"
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize_smiles(input_smiles))
    print(f"token_ids: {token_ids}")
    generated_sequences = []
    with torch.no_grad():
        if generate_config['num_beams'] > 1:
            generated_sequence = model.generate(inputs=torch.tensor(token_ids).unsqueeze(0).to(device),
                                                max_length=config_tokenizer["max_seq_len"],
                                                early_stopping=generate_config["early_stopping"],
                                                do_sample=generate_config["do_sample"],
                                                num_beam_groups=generate_config["num_beam_groups"],
                                                temperature=generate_config["temperature"],
                                                top_k=generate_config["top_k"],
                                                num_return_sequences=generate_config["num_return_sequences"],
                                                eos_token_id=tokenizer.eos_token_id,
                                                pad_token_id=tokenizer.pad_token_id,
                                                bos_token_id=tokenizer.bos_token_id,
                                                use_cache=False,)
            generated_sequences = [seq for seq in generated_sequence]
        else:
            for seq_no in tqdm(range(generate_config["num_return_sequences"])):
                if train_config["training"]["fp16"]:
                    input = torch.tensor(token_ids).unsqueeze(0).to(device)
                    input = torch.tile(input, (4, 1))
                    with autocast():
                        generated_sequence = model.generate(inputs=input,
                                                            max_length=config_tokenizer["max_seq_len"],
                                                            do_sample=generate_config["do_sample"],
                                                            early_stopping=generate_config["early_stopping"],
                                                            num_beam_groups=generate_config["num_beam_groups"],
                                                            temperature=generate_config["temperature"],
                                                            top_k=generate_config["top_k"],
                                                            eos_token_id=tokenizer.eos_token_id,
                                                            pad_token_id=tokenizer.pad_token_id,
                                                            bos_token_id=tokenizer.bos_token_id,
                                                            use_cache=True,)
                        # print(f"generated_sequence: {generated_sequence}")
                else:
                    generated_sequence = model.generate(inputs=torch.tensor(token_ids).unsqueeze(0).to(device),
                                                        max_length=config_tokenizer["max_seq_len"],
                                                        do_sample=generate_config["do_sample"],
                                                        early_stopping=generate_config["early_stopping"],
                                                        num_beam_groups=generate_config["num_beam_groups"],
                                                        temperature=generate_config["temperature"],
                                                        top_k=generate_config["top_k"],
                                                        eos_token_id=tokenizer.eos_token_id,
                                                        pad_token_id=tokenizer.pad_token_id,
                                                        bos_token_id=tokenizer.bos_token_id,
                                                        use_cache=True,)
                for seq in generated_sequence:
                    generated_sequences.append(seq)
            torch.cuda.empty_cache()

    valid_sequences = []
    sep_sign = "&&"
    for i, sequence in enumerate(generated_sequences):
        sequence_list = tokenizer.convert_ids_to_tokens(list(sequence.cpu().numpy().reshape(-1)))
        sequence_str = ''.join(sequence_list).replace("[PAD]", "").replace("[BOS]", "").replace("[MASK]", "").replace("[UNK]", "").strip()
        print(f"sequence {i}: {sequence_str}")
        if sep_sign in sequence_str:
            # number of separator tokens
            sep_count = sequence_str.count(sep_sign)
            if sep_count == 1:
                valid_sequences.append(sequence_str)
    
    print(f"valid_sequences count: {len(valid_sequences)} out of {len(generated_sequences)}")

if __name__ == "__main__":
    main()
        