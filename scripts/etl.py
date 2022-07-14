import torch
import pandas as pd
import numpy as np
import argparse

from utils.utils import parse_config, configure_logger


'''
Script that takes in the raw fer2013 csv file and outputs train/val/test datasets as pytorch tensors

Inputs:
- import_path [str] - path to the fer2013 dataset
- output_path [str] - path where train/val/test datasets will be saved

Outputs:
- saves train/val/test [pt] sets as pytorch tensors in the output_path
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfp', '--config_file_path', type=str, default='./scripts/config.yml', 
                         help='Path to the YAML configuration file')
    return parser.parse_args()



def etl():
    ######################
    # Configure logger
    ######################
    logger = configure_logger(__file__, 'log/etl.log')

    ######################
    # extract arguments from parser
    ######################
    args = parse_args()
    config = parse_config(args.config_file_path)
    RAW_DATA_PATH = config['etl']['raw_data_path']
    OUTPUT_FOLDER_PATH = config['etl']['output_folder_path']

    ######################
    # load and process dataset
    ######################
    logger.info('----------------------- Start ETL -----------------------')
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f'Data loaded from: {RAW_DATA_PATH}')

    pixels = df['pixels'].apply(lambda x: x.split(" ")).tolist()
    pixels = torch.Tensor(np.uint8(pixels))
    pixels = torch.reshape(pixels, (pixels.shape[0], 1, 48, 48))
    logger.info(f'Image data transformed to pytorch tensor | data type: {type(pixels)}, data shape: {pixels.shape}')

    ######################
    # train \ val \ test split
    ######################
    train_idx = df.index[df['Usage'] == 'Training'].tolist()
    val_idx   = df.index[df['Usage'] == 'PublicTest'].tolist()
    test_idx  = df.index[df['Usage'] == 'PrivateTest'].tolist()

    X_train = pixels[train_idx]
    y_train = torch.Tensor(df.iloc[train_idx]['emotion'].tolist())

    X_val = pixels[val_idx]
    y_val = torch.Tensor(df.iloc[val_idx]['emotion'].tolist())

    X_test = pixels[test_idx]
    y_test = torch.Tensor(df.iloc[test_idx]['emotion'].tolist())

    logger.info(f'Data split to train/val/test sets')
    logger.info(f'Train set shape: {X_train.shape}, val test shape: {X_val.shape}, test set shape: {X_test.shape}')

    ######################
    # save datasets
    ######################
    torch.save(X_train, OUTPUT_FOLDER_PATH + 'train_img.pt')
    torch.save(y_train, OUTPUT_FOLDER_PATH + 'train_labels.pt')
    torch.save(X_val,   OUTPUT_FOLDER_PATH + 'val_img.pt')
    torch.save(y_val,   OUTPUT_FOLDER_PATH + 'val_labels.pt')
    torch.save(X_test,  OUTPUT_FOLDER_PATH + 'test_img.pt')
    torch.save(y_test,  OUTPUT_FOLDER_PATH + 'test_labels.pt')
    logger.info(f'Files saved in {OUTPUT_FOLDER_PATH}')

if __name__ == '__main__':
    etl()
