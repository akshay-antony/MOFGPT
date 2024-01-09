from ast import mod
import sys
sys.path.append('..')
import torch
import yaml
import os
import glob
import sys
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataset.dataset_gpt import MOF_ID_Dataset
from tokenizer.mof_tokenizer import MOFTokenizer
from tokenizer.mof_tokenizer_gpt import MOFTokenizerGPT
from utils.data_utils import split_csv
from models.get_model import get_model
from transformers import GPT2Config, \
                         GPT2LMHeadModel, \
                         LlamaConfig, \
                         LlamaForCausalLM
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.functional.classification import multiclass_accuracy
import wandb
from models.models import LLMModel


def train_one_epoch(model,
                    model_class,
                    train_dataloader,
                    optimizer,
                    scheduler,
                    device,
                    epoch,
                    is_fp16,
                    logging_steps,
                    gradient_accumulation_steps,
                    scaler,
                    model_name,
                    save_dir,
                    save_steps,
                    config_model):
    model.train()
    loop = tqdm(train_dataloader,
                desc=f"Training Epoch {epoch}",
                colour='green',
                total=len(train_dataloader))
    total_train_loss = 0
    total_train_data = 0
    optimizer.zero_grad()

    for b_no, batch in enumerate(loop):
        token_ids = batch['token_ids'].to(device)
        mask_ids = batch['mask_ids'].to(device)
        target_token_ids = batch['target_token_ids'].to(device)
        outputs = model_class.forward(token_ids,
                                      mask_ids,
                                      target_token_ids,
                                      is_fp16)
        # if is_fp16:
        #     with autocast():
        #         outputs = model(input_ids=token_ids,
        #                         attention_mask=mask_ids,
        #                         labels=target_token_ids,
        #                         use_cache=config_model['use_cache'],
        #                         return_dict=config_model['return_dict'],
        #                         output_attentions=config_model['output_attentions'],
        #                         output_hidden_states=config_model['output_hidden_states'])
        #         loss = outputs.loss
        #         scaler.scale(loss).backward()
        # else:
        #     outputs = model(input_ids=token_ids,
        #                     attention_mask=mask_ids,
        #                     labels=target_token_ids,
        #                     use_cache=True)
        loss = outputs.loss
        if is_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        total_train_loss += loss.item() * token_ids.shape[0]
        total_train_data += token_ids.shape[0]

        if b_no % gradient_accumulation_steps == 0 and b_no != 0:
            if is_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        

        if b_no % logging_steps == 0 and b_no != 0:
            loop.set_postfix(loss=total_train_loss/total_train_data,
                             lr=scheduler.get_last_lr()[0])
            wandb.log({"train_loss_step": total_train_loss/total_train_data,
                       "lr": scheduler.get_last_lr()[0],
                       "step": epoch*len(train_dataloader) + b_no})
           
        if b_no % save_steps == 0:
            save_model(model,
                       epoch,
                       os.path.join(save_dir, f"{model_name}_latest.pt"),
                       optimizer,
                       scheduler,
                       loss)

    return total_train_loss/total_train_data

def eval_one_epoch(model,
                   model_class,
                   test_dataloader,
                   device,
                   epoch,
                   logging_steps=100,
                   is_fp16=False):
    model.eval()
    loop = tqdm(test_dataloader,
                desc=f"Evaluation Epoch {epoch}",
                colour='green',
                total=len(test_dataloader))
    total_test_loss = 0
    total_test_data = 0
    for b_no, batch in enumerate(loop):
        token_ids = batch['token_ids'].to(device)
        mask_ids = batch['mask_ids'].to(device)
        target_token_ids = batch['target_token_ids'].to(device)
        
        outputs = model_class.forward(token_ids,
                                      mask_ids,
                                      target_token_ids,
                                      is_fp16)
        loss = outputs.loss
        total_test_loss += loss.item() * token_ids.shape[0]
        total_test_data += token_ids.shape[0]
   

        if b_no % logging_steps == 0 and b_no != 0:
            loop.set_postfix(loss=total_test_loss/total_test_data)
    
    return total_test_loss/total_test_data

def save_model(model, 
               epoch, 
               save_path,
               optimizer,
               scheduler,
               loss):
    save_dict = {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(),
                 'loss': loss}
    torch.save(save_dict, save_path)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--config_filename", 
                      default="../config/config.yaml", 
                      type=str,
                      help="Path to config file")
    args = args.parse_args()

    with open(args.config_filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(config['model']['model_config_filename'], 'r') as f:   
        config_model = yaml.load(f, Loader=yaml.FullLoader)

    config_data = config['data']
    config_tokenizer = config_data['tokenizer']
    config_model = config_model[config['model']['model_name']]
    # config_model = config['model']
    config_training = config['training']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=config['project_name'],
               config=config)

    tokenizer = MOFTokenizerGPT(vocab_file=config['data']['vocab_path'],
                                add_special_tokens=config_tokenizer['add_special_tokens'],
                                truncation=config_tokenizer['truncation'],
                                pad_token=config_tokenizer['pad_token'],
                                mask_token=config_tokenizer['mask_token'],
                                bos_token=config_tokenizer['bos_token'],
                                eos_token=config_tokenizer['eos_token'],
                                unk_token=config_tokenizer['unk_token'],
                                max_len=config_tokenizer['max_seq_len'],
                                use_topology=config_tokenizer['use_topology'],)

    config_model['vocab_size'] = tokenizer.vocab_size
    print(f"tokenizer vocab size: {tokenizer.vocab_size}")

    # return
    csv_filenames = []
    for csv_folder_path in config['data']['csv_folder_paths']:
        csv_filenames.extend(glob.glob(csv_folder_path + '**/*.csv',
                                       recursive=True))
    print(f"Number of csv files: {len(csv_filenames)}")
    train_data_np, \
    test_data_np = split_csv(csv_filenames,
                             train_test_ratio = config['data']['train_test_ratio'],
                             random_seed = config['random_seed'])

    print("For train dataset:")
    train_dataset = MOF_ID_Dataset(train_data_np, 
                                   tokenizer,
                                   config_data['ignore_index'],
                                   use_multiprocessing=config_data['use_multiprocessing'])
    print("For test dataset:")
    test_dataset = MOF_ID_Dataset(test_data_np, 
                                  tokenizer,
                                  config_data['ignore_index'],
                                  use_multiprocessing=config_data['use_multiprocessing'])
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
    # Model config
    # adding special token ids to model config
    config_model['pad_token_id'] = tokenizer.pad_token_id
    config_model['bos_token_id'] = tokenizer.bos_token_id
    config_model['eos_token_id'] = tokenizer.eos_token_id
    config_model['vocab_size'] = tokenizer.vocab_size
    config_model['max_position_embeddings'] = config_tokenizer['max_seq_len']
    config_model['ignore_index'] = config_data['ignore_index']

    model = get_model(config['model']['model_name'],
                      config_model,
                      device)
    model_class = LLMModel(network=model,
                           model_name=config['model']['model_name'],
                           config=config_model)

    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6}M")

    # optimizer
    optimizer = getattr(torch.optim,
                        config_training['optimizer']['type'])
    optimizer = optimizer(model.parameters(), 
                          lr=config_training['optimizer']['lr'],
                          weight_decay=config_training['optimizer']['weight_decay'])
    # scheduler
    if config_training['scheduler']['type'] == 'cosine':
        num_training_steps = (len(train_dataloader) / config_training['optimizer']['gradient_accumulation_steps']) * config_training['epochs']
        num_warmup_steps = int(num_training_steps * config_training['scheduler']['warmup_ratio'])
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=num_warmup_steps,
                                                      num_training_steps=num_training_steps)
        print(f"created cosine scheduler with num_training_steps: {num_training_steps}, num_warmup_steps: {num_warmup_steps}")
    scaler = GradScaler() if config_training['fp16'] else None

    # making save dir
    if not os.path.exists(config['training']['save_dir']):
        os.makedirs(config['training']['save_dir'])

    min_test_loss = np.inf

    # calculating logging and save steps from ratios
    config_training['logging_steps'] = int(len(train_dataloader) * config_training['logging_ratio'])
    config_training['save_steps'] = int(len(train_dataloader) * config_training['save_ratio'])
    print(f"logging steps: {config_training['logging_steps']}, save steps: {config_training['save_steps']}")

    # training
    for epoch in range(config_training['epochs']):
        # training
        train_loss = train_one_epoch(model,
                                     model_class,
                                     train_dataloader,
                                     optimizer,
                                     scheduler,
                                     device,
                                     epoch,
                                     is_fp16=config_training['fp16'],
                                     logging_steps=config_training['logging_steps'],
                                     gradient_accumulation_steps=config_training['optimizer']['gradient_accumulation_steps'],
                                     scaler=scaler,
                                     model_name=config['model']['model_name'],
                                     save_dir=config['training']['save_dir'],
                                     save_steps=config_training['save_steps'],
                                     config_model=config_model)

        # evaluation
        test_loss = eval_one_epoch(model,
                                   model_class,
                                   test_dataloader,
                                   device,
                                   epoch,
                                   logging_steps=config_training['logging_steps'],
                                   is_fp16=config_training['fp16'])
        # save model if test loss is minimum
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            save_model(model,
                       epoch,
                       os.path.join(config['training']['save_dir'], f"{config['model']['model_name']}_best.pt"),
                       optimizer,
                       scheduler,
                       test_loss)
            print(f"Succeed saving model with test loss: {test_loss}")
        print(f"Epoch {epoch} train loss: {train_loss}, test loss: {test_loss}")
        # for train_acc, test_acc in zip(train_topks, test_topks):
        wandb.log({"epoch_train_loss": train_loss,
                    "epoch_test_loss": test_loss,
                    "epoch": epoch})
    return


if __name__ == "__main__":
    main()