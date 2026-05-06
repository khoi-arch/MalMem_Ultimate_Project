import os
import gc
import math
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from pathlib import Path
import warnings
import concurrent.futures
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")

# ==========================================
# 1. CẤU HÌNH TĨNH & HYPERPARAMS
# ==========================================
class Config:
    # BẬT 4 SEED ĐỂ CHẠY CÂN BẰNG 50-50 TRÊN 2 GPU
    SEEDS = [42, 2024, 3407, 8888]       
    BATCH_SIZE = 256
    EPOCHS = 100       
    PATIENCE = 15
    LR = 2e-4
    WEIGHT_DECAY = 1e-2
    SUPCON_WEIGHT = 0.1
    SUPCON_TEMP = 0.07
    MAX_TOKENS = 256               

    PROJECT_ROOT = Path(__file__).resolve().parent
    PROCESSED_DIR = PROJECT_ROOT / "data" / "split_80_20" / "processed"
    METADATA_DIR = PROJECT_ROOT / "data" / "split_80_20" / "metadata"
    TRAIN_CSV = str(PROCESSED_DIR / "train_processed.csv")
    VAL_CSV = str(PROCESSED_DIR / "val_processed.csv")
    GROUPS_JSON = str(METADATA_DIR / "feature_groups.json")
    TARGET_COL = "label_L3"

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & KIỂM SOÁT PURE CONTROL
# ==========================================
class ExpDataset(Dataset):
    def __init__(self, csv_path, is_train=True, label_encoder=None, ordered_features=None):
        df = pd.read_csv(csv_path)
        if ordered_features is not None:
            valid_ordered = [c for c in ordered_features if c in df.columns]
            X_df = df[valid_ordered]
        else:
            X_df = df.drop(columns=[Config.TARGET_COL])
            
        self.feature_names = X_df.columns.tolist()
        self.X = torch.tensor(X_df.values, dtype=torch.float32)
        y_raw = df[Config.TARGET_COL].values
        
        if is_train:
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.y = torch.tensor(self.label_encoder.fit_transform(y_raw), dtype=torch.long)
        else:
            self.label_encoder = label_encoder
            self.y = torch.tensor(self.label_encoder.transform(y_raw), dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def load_data(fga_mode):
    ordered_features, group_counts = None, []
    
    if fga_mode != 'baseline':
        if not os.path.exists(Config.GROUPS_JSON):
            raise FileNotFoundError(f"Missing {Config.GROUPS_JSON}.")
            
        with open(Config.GROUPS_JSON, 'r') as f:
            groups_data = json.load(f)["groups_summary"]
            
        original_ordered, original_counts = [], []
        for g_name, g_info in groups_data.items():
            features_in_group = list(g_info["features"].keys())
            original_ordered.extend(features_in_group)
            original_counts.append(len(features_in_group))
            
        if fga_mode == 'random_fixed_size':
            all_features = original_ordered.copy()
            random.shuffle(all_features)
            group_counts = original_counts
            ordered_features = all_features
            
        elif fga_mode == 'random_random_size':
            all_features = original_ordered.copy()
            random.shuffle(all_features)
            total_features = len(all_features)
            group_counts, ordered_features, idx = [], [], 0
            while idx < total_features:
                size = random.randint(2, min(10, total_features - idx))
                if total_features - (idx + size) == 1: size += 1 
                group_counts.append(size)
                ordered_features.extend(all_features[idx:idx+size])
                idx += size
                
        elif fga_mode == 'shuffle_within':
            ordered_features = []
            group_counts = original_counts
            for g_name, g_info in groups_data.items():
                features = list(g_info["features"].keys())
                random.shuffle(features)
                ordered_features.extend(features)
        else:
            ordered_features = original_ordered
            group_counts = original_counts
            
    train_ds = ExpDataset(Config.TRAIN_CSV, is_train=True, ordered_features=ordered_features)
    val_ds = ExpDataset(Config.VAL_CSV, is_train=False, label_encoder=train_ds.label_encoder, ordered_features=ordered_features)
    
    # BẮT BUỘC num_workers=0 để chống giật LAG CPU và Deadlock
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)
    
    counts = np.bincount(train_ds.y.numpy())
    class_priors = torch.tensor(counts / counts.sum(), dtype=torch.float32)
    
    if fga_mode == 'baseline': group_counts = [len(train_ds.feature_names)]
    return train_loader, val_loader, train_ds.label_encoder, group_counts, class_priors

# ==========================================
# 3. KHO VŨ KHÍ: LOSS FUNCTIONS 
# ==========================================
class LogitAdjustedLoss(nn.Module):
    def __init__(self, class_priors, tau=1.0, rare_boost=False):
        super().__init__()
        tau_tensor = torch.full_like(class_priors, tau)
        if rare_boost:
            bottom_3_idx = torch.argsort(class_priors)[:3]
            tau_tensor[bottom_3_idx] *= 1.5 
        self.adjustments = (tau_tensor * torch.log(class_priors + 1e-9))
        
    def forward(self, logits, labels):
        adj = self.adjustments.to(logits.device)
        return F.cross_entropy(logits + adj, labels)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        device = features.device 
        features = F.normalize(features, dim=1)
        similarity = torch.div(torch.matmul(features, features.T), self.temperature)
        labels = labels.contiguous().view(-1, 1)
        
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask - torch.eye(mask.shape[0]).to(device)
        
        max_sim, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - max_sim.detach()
        exp_logits = torch.exp(logits) * (1 - torch.eye(mask.shape[0]).to(device))
        
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        return -mean_log_prob_pos.mean()

# ==========================================
# 4. KIẾN TRÚC MẠNG NEURAL 
# ==========================================
class SinCosTokenizer(nn.Module):
    def __init__(self, num_features, d_model, k=4):
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(num_features, k))
        self.mlp = nn.Linear(k * 2, d_model)
        self.column_embeddings = nn.Parameter(torch.randn(1, num_features, d_model) * 0.02)
        
    def forward(self, x):
        angles = 2 * math.pi * x.unsqueeze(-1) * self.frequencies
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) 
        return self.mlp(emb) + self.column_embeddings.expand(x.size(0), -1, -1)

class ExperimentModel(nn.Module):
    def __init__(self, group_counts, num_classes, fga_mode='baseline', use_supcon=False, embed_dim=128):
        super().__init__()
        self.group_counts = group_counts
        self.fga_mode = fga_mode  
        self.use_supcon = use_supcon
        
        if self.fga_mode != 'baseline':
            self.tokenizers = nn.ModuleList([SinCosTokenizer(cnt, embed_dim) for cnt in group_counts])
            self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in group_counts])
            
            if self.fga_mode == 'attn':
                self.attn_pools = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(embed_dim, embed_dim // 2),
                        nn.ReLU(),
                        nn.Linear(embed_dim // 2, 1),
                        nn.Softmax(dim=1)
                    ) for _ in group_counts
                ])
            elif self.fga_mode == 'attn_linear':
                self.attn_pools = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(embed_dim, 1),
                        nn.Softmax(dim=1)
                    ) for _ in group_counts
                ])
        else:
            self.tokenizer = SinCosTokenizer(sum(group_counts), embed_dim)
            self.norm = nn.LayerNorm(embed_dim)
            
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier_head = nn.Linear(embed_dim, num_classes)
        
        if self.use_supcon:
            self.projection_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 64))
            
    def forward(self, x):
        if self.fga_mode != 'baseline':
            x_splits = torch.split(x, self.group_counts, dim=1)
            group_tokens = []
            for i in range(len(self.group_counts)):
                toks = self.norms[i](self.tokenizers[i](x_splits[i]))
                if self.fga_mode in ['mean', 'random_fixed_size', 'random_random_size', 'shuffle_within']:
                    group_tokens.append(toks.mean(dim=1, keepdim=True))
                elif self.fga_mode in ['attn', 'attn_linear']:
                    weights = self.attn_pools[i](toks)
                    group_tokens.append((toks * weights).sum(dim=1, keepdim=True))
                elif self.fga_mode in ['no_pool_rand', 'no_pool_topk']:
                    group_tokens.append(toks) 
            tokens = torch.cat(group_tokens, dim=1) 
        else:
            tokens = self.norm(self.tokenizer(x))
            
        if tokens.shape[1] > Config.MAX_TOKENS:
            if self.fga_mode == 'no_pool_topk':
                var = tokens.detach().var(dim=-1) 
                topk_idx = torch.topk(var, k=Config.MAX_TOKENS, dim=1).indices
                topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
                tokens = torch.gather(tokens, 1, topk_idx_expanded)
            else:
                if self.training:
                    idx = torch.randperm(tokens.shape[1], device=tokens.device)[:Config.MAX_TOKENS]
                    tokens = tokens[:, idx, :]
                else:
                    idx = torch.linspace(0, tokens.shape[1] - 1, steps=Config.MAX_TOKENS, dtype=torch.long, device=tokens.device)
                    tokens = tokens[:, idx, :]
                
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x_trans = torch.cat([cls, tokens], dim=1)
        x_trans = self.transformer(x_trans)
        
        cls_output = x_trans[:, 0]
        logits = self.classifier_head(cls_output)
        
        if self.use_supcon: 
            mean_feature_tokens = x_trans[:, 1:].mean(dim=1)
            proj = self.projection_head(mean_feature_tokens)
            return logits, proj
            
        return logits, None

# ==========================================
# 5. ENGINE: TRAIN & ĐÁNH GIÁ (GPU CHỈ ĐỊNH)
# ==========================================
def run_single_seed(seed, exp_config, gpu_id=0):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        
    seed_everything(seed)
    
    tau = exp_config.get("tau", None)
    rare_boost = exp_config.get("rare_boost", False)
    fga_mode = exp_config.get("fga", "baseline")
    supcon_mode = exp_config.get("supcon", False)
    
    train_loader, val_loader, encoder, group_counts, class_priors = load_data(fga_mode)
    num_classes = len(encoder.classes_)
    
    model = ExperimentModel(group_counts, num_classes, fga_mode, bool(supcon_mode), embed_dim=128).to(device)
    
    if tau is not None: criterion_ce = LogitAdjustedLoss(class_priors, tau=tau, rare_boost=rare_boost)
    else: criterion_ce = nn.CrossEntropyLoss()
        
    criterion_supcon = SupConLoss(temperature=Config.SUPCON_TEMP) if supcon_mode else None
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    best_val_f1, patience_counter = 0.0, 0
    best_preds, best_labels = [], []
    best_report_dict = {}
    running_sc_mean = 0.0  
    
    for epoch in range(Config.EPOCHS):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, proj = model(x)
            loss_ce = criterion_ce(logits, y)
            
            if supcon_mode:
                loss_sc = criterion_supcon(proj, y)
                sc_val = loss_sc.detach().item()
                
                if epoch == 0 and step == 0: running_sc_mean = sc_val
                else: running_sc_mean = 0.9 * running_sc_mean + 0.1 * sc_val
                
                if supcon_mode == 'S1': loss = loss_ce + Config.SUPCON_WEIGHT * loss_sc
                elif supcon_mode == 'S2': loss = loss_ce + Config.SUPCON_WEIGHT * (loss_sc / (running_sc_mean + 1e-6))
                elif supcon_mode == 'S3': 
                    if epoch < int(0.2 * Config.EPOCHS): loss = loss_sc 
                    elif epoch < int(0.6 * Config.EPOCHS): loss = loss_ce + Config.SUPCON_WEIGHT * loss_sc 
                    else: loss = loss_ce 
                elif supcon_mode == 'S4':
                    if epoch < int(0.2 * Config.EPOCHS): loss = loss_sc 
                    elif epoch < int(0.6 * Config.EPOCHS): loss = loss_ce + Config.SUPCON_WEIGHT * loss_sc 
                    else: loss = loss_ce + 0.05 * loss_sc 
            else:
                loss = loss_ce
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        # --- ĐÁNH GIÁ CUỐI EPOCH ---
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits, _ = model(x.to(device))
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                labels.extend(y.numpy())
        
        val_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        scheduler.step(val_f1)
        
        is_best = False
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_preds, best_labels = preds, labels
            from sklearn.metrics import classification_report
            best_report_dict = classification_report(labels, preds, target_names=encoder.classes_, digits=4, zero_division=0, output_dict=True)
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1
            
        # =======================================================
        # [HEARTBEAT LOGGING] - IN TIẾN ĐỘ THÔNG MINH
        # Chỉ in ra khi có kỷ lục mới HOẶC mỗi 5 Epochs 
        # =======================================================
        if is_best or epoch % 20 == 0:
            marker = "🔥 NEW BEST" if is_best else "⏳ Running"
            print(f"      [GPU:{gpu_id} | Seed:{seed} | E{epoch:02d}] Val F1: {val_f1:.4f} ({marker})")

        if patience_counter >= Config.PATIENCE: 
            print(f"      [GPU:{gpu_id} | Seed:{seed}] 🛑 Early Stopping ở Epoch {epoch:02d}")
            break
            
    weighted_f1 = f1_score(best_labels, best_preds, average='weighted', zero_division=0)
    bottom_3_idx = torch.argsort(class_priors)[:3].tolist()
    rare_f1_scores = {encoder.classes_[i]: round(best_report_dict[encoder.classes_[i]]['f1-score'], 4) for i in bottom_3_idx} if best_report_dict else {}
            
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    return best_val_f1, weighted_f1, rare_f1_scores

# ==========================================
# 6. ORCHESTRATOR: KẾT NỐI ĐA GPU 
# ==========================================
def run_experiment_multiseed(exp_id, exp_name, exp_config):
    num_gpus = torch.cuda.device_count()
    # IN RA SỐ LƯỢNG TIẾN TRÌNH
    print(f"\n[>] ĐANG CHẠY: {exp_id} - {exp_name} (SỬ DỤNG {num_gpus} GPU - {len(Config.SEEDS)} SEEDS CÙNG LÚC)")
    macro_scores, weighted_scores, rare_records = [], [], []
    
    ctx = mp.get_context('spawn')
    # BẬT max_workers = 4 (Ép xung toàn diện)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=ctx) as executor:
        futures = {}
        for i, seed in enumerate(Config.SEEDS):
            gpu_id = i % num_gpus if num_gpus > 0 else 0
            futures[executor.submit(run_single_seed, seed, exp_config, gpu_id)] = seed
        
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                macro, weighted, rare = future.result()
                macro_scores.append(macro)
                weighted_scores.append(weighted)
                rare_records.append(rare)
                print(f"    ✅ Hoàn thành Seed {seed} -> Final F1: {macro:.4f}")
            except Exception as e:
                print(f"    ❌ [LỖI NGHIÊM TRỌNG] Seed {seed} crash: {str(e)}")
            
    if not macro_scores:
        return 0.0, 0.0, 0.0, {}
        
    mean_macro, std_macro = np.mean(macro_scores), np.std(macro_scores)
    mean_weighted = np.mean(weighted_scores)
    
    avg_rare = {}
    if rare_records and rare_records[0]:
        for key in rare_records[0].keys(): 
            avg_rare[key] = round(np.mean([r[key] for r in rare_records]), 4)
        
    print(f"    => TỔNG KẾT EXPERIMENT: Macro F1 = {mean_macro:.4f} ± {std_macro:.4f}")
    return mean_macro, std_macro, mean_weighted, avg_rare

# ==========================================
# 7. MAIN ENTRY
# ==========================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) 
    
    experiments = [
        # A. FGA Variants
        {"id": "E2-F2a","name": "FGA: Attn Pooling (Linear)",         "fga": "attn_linear",      "supcon": False},
        {"id": "E2-F2b","name": "FGA: Attn Pooling (MLP)",            "fga": "attn",             "supcon": False},
        {"id": "E2-F3", "name": "FGA: No Pool (Rand Drop/Linspace)",  "fga": "no_pool_rand",     "supcon": False},
        {"id": "E2-F3b","name": "FGA: No Pool (Top-K Var Detach)",    "fga": "no_pool_topk",     "supcon": False},
        {"id": "E2-F4a","name": "FGA Control: Fixed Size Random Feat","fga": "random_fixed_size","supcon": False},
        {"id": "E2-F4b","name": "FGA Control: Random Size & Feat",    "fga": "random_random_size","supcon": False},
        {"id": "E2-F5", "name": "FGA Sanity: Shuffle Within Group",   "fga": "shuffle_within",   "supcon": False},
        
        # B. SupCon Variants
        {"id": "E3-S1", "name": "SupCon: Naive",                      "fga": "baseline", "supcon": "S1"},
        {"id": "E3-S2", "name": "SupCon: EMA Normalized",             "fga": "baseline", "supcon": "S2"},
        {"id": "E3-S3", "name": "SupCon: 3-Phase Warmup (Drop)",      "fga": "baseline", "supcon": "S3"},
        {"id": "E3-S4", "name": "SupCon: 3-Phase Aligned (0.05)",     "fga": "baseline", "supcon": "S4"},
        
        # C. LA Variants
        {"id": "E1-L1", "name": "LA: Tau = 1.0",                      "fga": "baseline", "supcon": False, "tau": 1.0, "rare_boost": False},
        {"id": "E1-L2", "name": "LA: Tau = 1.5",                      "fga": "baseline", "supcon": False, "tau": 1.5, "rare_boost": False},
        {"id": "E1-L3", "name": "LA: Tau = 1.0 + Rare Boost",         "fga": "baseline", "supcon": False, "tau": 1.0, "rare_boost": True},
    ]
    
    results = []
    print("🚀 BẮT ĐẦU ABLATION STUDY V7 (FULL DUAL-GPU / 4 SEEDS ENGINE)")
    for exp in experiments:
        mean_macro, std_macro, mean_weighted, avg_rare = run_experiment_multiseed(exp["id"], exp["name"], exp)
        results.append({
            "Exp ID": exp["id"],
            "Config": exp["name"], 
            "Macro F1 (4 Seeds)": f"{mean_macro:.4f} ± {std_macro:.4f}",
            "Weighted F1": round(mean_weighted, 4),
            "Rare Class F1": avg_rare,
            "Raw Mean": mean_macro
        })
        
    df_results = pd.DataFrame(results)
    
    if 'Rare Class F1' in df_results.columns and isinstance(df_results['Rare Class F1'].iloc[0], dict) and df_results['Rare Class F1'].iloc[0]:
        rare_df = df_results['Rare Class F1'].apply(pd.Series)
        df_final = pd.concat([df_results.drop(['Rare Class F1', 'Raw Mean'], axis=1), rare_df], axis=1)
    else:
        df_final = df_results.drop(columns=['Raw Mean'], errors='ignore')
    
    print("\n\n" + "★"*90)
    print("🏆 KẾT QUẢ PHÂN RÃ KỸ THUẬT (AUTO ABLATION REPORT V7)")
    print("★"*90)
    print(df_final.to_markdown(index=False))
    print("★"*90)
    
    csv_path = Config.PROJECT_ROOT / "ablation_results_kaggle.csv"
    df_final.to_csv(csv_path, index=False)
    print(f"\nĐã xuất kết quả ra file CSV: {csv_path}")