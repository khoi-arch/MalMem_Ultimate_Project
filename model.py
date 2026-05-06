import math
import torch
import torch.nn as nn

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

class UltimateHybridModel(nn.Module):
    def __init__(self, num_features, num_classes, embed_dim=128, dropout_rate=0.0):
        super().__init__()
        self.tokenizer = SinCosTokenizer(num_features, embed_dim)
        self.bridge_norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4, batch_first=True, norm_first=True, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        tokens = self.bridge_norm(self.tokenizer(x))
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = self.transformer(x)
        return self.head(x[:, 0])