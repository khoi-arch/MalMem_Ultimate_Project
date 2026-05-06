import os
import torch
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class MalwareDataset(Dataset):
    def __init__(self, csv_path, target_col='label_L3', label_encoder=None, encoder_save_path=None, train_columns=None):
        if not os.path.exists(csv_path): raise FileNotFoundError(f"Missing file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        X_df = df.drop(columns=[target_col])
        self.feature_names = X_df.columns.tolist()
        
        if train_columns is not None and self.feature_names != train_columns:
            raise ValueError("Feature mismatch between Train and Val/Test!")

        self.X = torch.tensor(X_df.values, dtype=torch.float32)
        y_raw = df[target_col].values
        
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            encoded_y = self.label_encoder.fit_transform(y_raw)
            if encoder_save_path:
                os.makedirs(os.path.dirname(encoder_save_path), exist_ok=True)
                joblib.dump(self.label_encoder, encoder_save_path)
        else:
            self.label_encoder = label_encoder
            unknown_mask = ~np.isin(y_raw, self.label_encoder.classes_)
            if unknown_mask.any(): raise ValueError(f"Unknown labels found: {np.unique(y_raw[unknown_mask])}")
            encoded_y = self.label_encoder.transform(y_raw)
            
        self.y = torch.tensor(encoded_y, dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def get_dataloaders(train_csv, val_csv, target_col, metadata_dir, batch_size=256):
    encoder_path = os.path.join(metadata_dir, "label_encoder.pkl")
    train_ds = MalwareDataset(csv_path=train_csv, target_col=target_col, encoder_save_path=encoder_path)
    val_ds = MalwareDataset(csv_path=val_csv, target_col=target_col, label_encoder=train_ds.label_encoder, train_columns=train_ds.feature_names)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, train_ds.label_encoder, train_ds.feature_names