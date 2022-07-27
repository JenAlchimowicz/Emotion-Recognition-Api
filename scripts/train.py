import torch
import argparse
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.utils import parse_config, configure_logger
from train_utils.create_dataloaders import create_dataloaders
from architectures.ED_model import ED_model


'''
Script that takes data and hyperparameters and trains a model

Inputs:
    - data_path [str] - path to the train/val/test set folder
    - epochs [int] - number of epochs to train on
    - batch size [int]
    - start_lr [float] - starting learning rate (training file uses dynamic learning rate so it will change as the training progresses)
    - model_path [str] - where the model should be saved

Outputs:
    - Model [pt] saved in the model_path file
    - Report [csv] with loss, accuracy and f1 for each epoch saved in the logs folder
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfp', '--config_file_path', type=str, default='./scripts/config.yml', 
                         help='Path to the YAML configuration file')
    return parser.parse_args()


def train():

    ######################
    # Configure logger
    ######################
    logger = configure_logger(__file__, 'log/train.log')

    ######################
    # Extract arguments from parser
    ######################
    args = parse_args()
    config = parse_config(args.config_file_path)
    DATA_PATH  = config['train']['data_path']
    EPOCHS     = config['train']['epochs']
    BATCH_SIZE = config['train']['batch_size']
    START_LR   = config['train']['start_lr']
    MODEL_PATH = config['train']['model_path']

    ######################
    # Create dataloaders
    ######################
    logger.info('----------------------- Start training script -----------------------')
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(BATCH_SIZE,
                                                                           torch.load(DATA_PATH+'train_img.pt'),
                                                                           torch.load(DATA_PATH+'train_labels.pt'),
                                                                           torch.load(DATA_PATH+'val_img.pt'),
                                                                           torch.load(DATA_PATH+'val_labels.pt'),
                                                                           torch.load(DATA_PATH+'test_img.pt'),
                                                                           torch.load(DATA_PATH+'test_labels.pt'))
    logger.info('Dataloaders created')

    ######################
    # Initialize training pipeline
    ######################
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = ED_model(in_channels=1, out_channels=7).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=START_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) # Use dynamic learning rate
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # Trackers
    total_train_losses   = []
    total_val_losses     = []
    total_train_accuracy = []
    total_val_accuracy   = []
    total_train_f1       = []
    total_val_f1         = []
    learning_rate_tracker = []
    logger.info(f'Model, optimizer, scheduler and criterion initialised, starting training')

    ######################
    # Train model
    ######################
    for epoch in range(1, EPOCHS+1):
    
        model.train()
        train_losses, train_accuracy, train_f1 = [], [], []
    
        for i, batch in enumerate(train_dataloader):
            img_batch, label_batch = batch   #img [B,3,H,W], label[B,N_CLASSES]
            img_batch = img_batch.to(DEVICE)
            label_batch = label_batch.type(torch.LongTensor).to(DEVICE)

            optimizer.zero_grad()
            output = model(img_batch) # output: [B, 7, H, W]
            loss = criterion(output, label_batch)
            loss.backward()
            optimizer.step()

            #Add current metrics to temporary lists (take average after full epoch)
            preds = torch.argmax(output, dim=1)
            f1 = f1_score(preds.cpu(), label_batch.cpu(), average='macro')
            acc = torch.sum(preds == label_batch) / len(preds)
            train_losses.append(loss.item())
            train_accuracy.append(acc.cpu())
            train_f1.append(f1)
    
        # Update global trackers
        print(f'TRAIN       Epoch: {epoch} | Epoch metrics | loss: {np.mean(train_losses):.4f}, f1: {np.mean(train_f1):.3f}, accuracy: {np.mean(train_accuracy):.3f}, learning rate: {optimizer.state_dict()["param_groups"][0]["lr"]:.6f}')        
        total_train_losses.append(np.mean(train_losses))
        total_train_accuracy.append(np.mean(train_accuracy))
        total_train_f1.append(np.mean(train_f1))
        
        #Update learning rate
        learning_rate_tracker.append(optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()
    
        # Validation set performance
        model.eval()
        val_losses, val_accuracy, val_f1 = [], [], []
    
        for i, batch in enumerate(val_dataloader):
            img_batch, label_batch = batch
            img_batch = img_batch.to(DEVICE)
            label_batch = label_batch.type(torch.LongTensor).to(DEVICE)

            with torch.cuda.amp.autocast():
                output = model(img_batch)
                loss   = criterion(output, label_batch)

            #Add current metrics to temporary lists (take average after full epoch)
            preds = torch.argmax(output, dim=1)
            f1 = f1_score(preds.cpu(), label_batch.cpu(), average='macro')
            acc = torch.sum(preds == label_batch) / len(preds)
            val_losses.append(loss.item())
            val_accuracy.append(acc.cpu())
            val_f1.append(f1)
    
        # Update global trackers
        print(f'VALIDATION  Epoch: {epoch} | Epoch metrics | loss: {np.mean(val_losses):.4f}, f1: {np.mean(val_f1):.3f}, accuracy: {np.mean(val_accuracy):.3f}')
        print('-'*106)
        total_val_losses.append(np.mean(val_losses))
        total_val_accuracy.append(np.mean(val_accuracy))
        total_val_f1.append(np.mean(val_f1))
    logger.info('Training complete | last epoch metrics | Training loss: {total_train_losses[-1]}, val loss: {total_val_losses[-1]}, training accuracy: {total_train_accuracy[-1]}, val accuracy: {total_val_accuracy[-1]}, training f1: {total_train_f1[-1]}, val f1: {total_val_f1[-1]}')

    
    # Save the model
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f'Model saved in {MODEL_PATH}')

    # Save the results
    temp_df = pd.DataFrame(list(zip(total_train_losses, total_val_losses, total_train_f1, total_val_f1,
                                    total_train_accuracy, total_val_accuracy)),
                            columns = ['train_loss', 'val_loss', 'train_f1', 'test_f1', 'train_accuracy',
                                    'test_accuracy'])
    temp_df.to_csv('log/results.csv')
    logger.info(f'Detailed results saved in log/results.csv')
      

if __name__ == '__main__':
    train()
