import torch
from pathlib import Path

class Config:
    # --- ĐƯỜNG DẪN TỰ ĐỘNG ---
    PROJECT_ROOT = Path(__file__).resolve().parent
    PROCESSED_DIR = PROJECT_ROOT / "data" / "split_80_20" / "processed"
    METADATA_DIR = PROJECT_ROOT / "data" / "split_80_20" / "metadata"
    TRAIN_CSV = str(PROCESSED_DIR / "train_processed.csv")
    VAL_CSV = str(PROCESSED_DIR / "val_processed.csv")
    
    TARGET_COL = "label_L3"
    
    # --- TRAINING PARAMETERS ---
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 256
    EPOCHS = 100
    PATIENCE = 15          
    
    # --- HYPERPARAMETERS ---
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2    
    DROPOUT_RATE = 0.0