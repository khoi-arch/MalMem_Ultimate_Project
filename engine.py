import torch
from sklearn.metrics import f1_score

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