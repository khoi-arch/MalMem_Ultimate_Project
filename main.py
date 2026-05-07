import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

from config import Config
from dataset import get_dataloaders
from model import UltimateHybridModel
from train import evaluate, train_one_epoch, LogitAdjustedLoss  # Chú ý import LogitAdjustedLoss từ train.py
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =====================================================================
# HÀM MỚI: TỰ ĐỘNG ĐỌC JSON, LẤY GROUP COUNTS & XUẤT MAPPING FILE
# =====================================================================
def load_fga_mapping(json_path):
    with open(json_path, 'r') as f:
        groups_data = json.load(f)["groups_summary"]
        
    ordered_features = []
    group_counts = []
    mapping_dict = {}
    
    for g_name, g_info in groups_data.items():
        features = list(g_info["features"].keys())
        ordered_features.extend(features)
        group_counts.append(len(features))
        mapping_dict[g_name] = {
            "feature_count": len(features),
            "features": features
        }
        
    # Xuất ra file JSON để ông dễ kiểm tra
    mapping_file = "fga_column_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(mapping_dict, f, indent=4)
        
    print(f"✅ Đã xuất sơ đồ phân nhóm FGA ra file: {mapping_file}")
    return ordered_features, group_counts

def run_pipeline():
    seed_everything(Config.SEED)
    print("🚀 TÍCH HỢP HỆ THỐNG: THE ULTIMATE COMBO (FGA MLP + LA)")
    
    # 1. Load cấu trúc FGA từ JSON
    ordered_features, group_counts = load_fga_mapping(Config.GROUPS_JSON)
    
    # 2. Load Data (LƯU Ý: Phải đảm bảo get_dataloaders sắp xếp cột theo ordered_features)
    # [SỬA QUAN TRỌNG TẠI ĐÂY] - Đã thêm ordered_features=ordered_features
    train_loader, val_loader, encoder, features = get_dataloaders(
        Config.TRAIN_CSV, 
        Config.VAL_CSV, 
        Config.TARGET_COL, 
        str(Config.METADATA_DIR), 
        Config.BATCH_SIZE,
        ordered_features=ordered_features 
    )
    num_classes = len(encoder.classes_)
    
    # 3. Tính toán Class Priors cho Logit Adjusted Loss (LA)
    # LA cần biết tỷ lệ phần trăm của từng class (priors) chứ không phải class_weights thông thường
    all_y = []
    for _, y in train_loader: 
        all_y.extend(y.numpy())
    counts = np.bincount(all_y, minlength=num_classes)
    class_priors = counts / counts.sum() # Tính tỷ lệ % của mỗi class
    class_priors_tensor = torch.tensor(class_priors, dtype=torch.float32).to(Config.DEVICE)
    
    # 4. Khởi tạo Model (Dùng group_counts thay vì len(features))
    model = UltimateHybridModel(group_counts=group_counts, num_classes=num_classes, dropout_rate=Config.DROPOUT_RATE).to(Config.DEVICE)
    
    # Khởi tạo Hàm Loss Trùm cuối (Logit Adjusted Loss với Tau=1.0)
    criterion = LogitAdjustedLoss(class_priors=class_priors_tensor, tau=1.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    best_val_f1, best_weights, patience_counter = 0.0, None, 0
    
    # 5. Training Loop
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

    # 6. Report
    model.load_state_dict(best_weights)
    test_f1, preds, labels = evaluate(model, val_loader, Config.DEVICE)
    
    print("\n" + "="*60)
    print(f"🏆 TEST REPORT (ULTIMATE COMBO - VAL SET)")
    print(f"F1 Macro: {test_f1:.4f}")
    print(classification_report(labels, preds, target_names=encoder.classes_, digits=4, zero_division=0))

if __name__ == "__main__":
    run_pipeline()