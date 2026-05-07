import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

class LogitAdjustedLoss(nn.Module):
    def __init__(self, class_priors, tau=1.0):
        super().__init__()
        # class_priors: tensor chứa tỉ lệ phân bố của các class trong tập Train
        # tau: tham số kiểm soát độ phạt (default = 1.0 là tối ưu nhất)
        tau_tensor = torch.full_like(class_priors, tau)
        self.adjustments = (tau_tensor * torch.log(class_priors + 1e-9))
        
    def forward(self, logits, labels):
        # Cộng thêm trọng số phạt vào logits trước khi tính Cross Entropy
        adj = self.adjustments.to(logits.device)
        return F.cross_entropy(logits + adj, labels)

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.numpy())
    return f1_score(labels, preds, average='macro', zero_division=0), preds, labels

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)