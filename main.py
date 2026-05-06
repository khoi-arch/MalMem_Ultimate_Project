import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

from config import Config
from dataset import get_dataloaders
from model import UltimateHybridModel
from engine import evaluate, train_one_epoch
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_pipeline():
    seed_everything(Config.SEED)
    print("🚀 TÍCH HỢP HỆ THỐNG: RESEARCH-GRADE PIPELINE")
    
    # 1. Load Data từ thư mục Processed
    train_loader, val_loader, encoder, features = get_dataloaders(
        Config.TRAIN_CSV, Config.VAL_CSV, Config.TARGET_COL, str(Config.METADATA_DIR), Config.BATCH_SIZE
    )
    
    num_classes = len(encoder.classes_)
    
    # Cân bằng Class Weights
    all_y = []
    for _, y in train_loader: all_y.extend(y.numpy())
    counts = np.bincount(all_y, minlength=num_classes)
    weights = len(all_y) / (num_classes * np.where(counts == 0, 1, counts))
    class_weights = torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)
    
    # 2. Khởi tạo Model
    model = UltimateHybridModel(len(features), num_classes, dropout_rate=Config.DROPOUT_RATE).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    best_val_f1, best_weights, patience_counter = 0.0, None, 0
    
    # 3. Training
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_f1, _, _ = evaluate(model, val_loader, Config.DEVICE)
        scheduler.step(val_f1)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.2e}")
        
        if val_f1 > best_val_f1: 
            best_val_f1 = val_f1
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else: 
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"⛔ Early Stopping kích hoạt ở Epoch {epoch+1}!")
                break

    # 4. Report
    model.load_state_dict(best_weights)
    test_f1, preds, labels = evaluate(model, val_loader, Config.DEVICE)
    
    print("\n" + "="*60)
    print(f"🏆 TEST REPORT (VAL SET)")
    print(f"F1 Macro: {test_f1:.4f}")
    print(classification_report(labels, preds, target_names=encoder.classes_, digits=4, zero_division=0))

if __name__ == "__main__":
    run_pipeline()