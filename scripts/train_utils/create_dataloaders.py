from torch.utils.data import DataLoader

from train_utils.dataloader import EDtensorDataset

def create_dataloaders(BATCH_SIZE, X_train, y_train, X_val, y_val, X_test=None, y_test=None):

    train_dataset = EDtensorDataset(X_train, y_train, train=True)
    val_dataset   = EDtensorDataset(X_val, y_val, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader   = DataLoader(val_dataset  , batch_size=BATCH_SIZE, shuffle=False)

    if X_test is not None:
        test_dataset  = EDtensorDataset(X_test, y_test, train=False)
        test_dataloader  = DataLoader(test_dataset , batch_size=BATCH_SIZE, shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader
    
    return train_dataloader, val_dataloader