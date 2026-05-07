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
    def __init__(self, group_counts, num_classes, embed_dim=128, dropout_rate=0.0):
        super().__init__()
        self.group_counts = group_counts
        
        # 1. Tokenizers và Normalization độc lập cho từng nhóm
        self.tokenizers = nn.ModuleList([SinCosTokenizer(cnt, embed_dim) for cnt in group_counts])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in group_counts])
        
        # 2. FGA: Attention Pooling (MLP) - Tự động học trọng số để lọc nhiễu
        self.attn_pools = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Softmax(dim=1)
            ) for _ in group_counts
        ])
        
        # 3. Khởi tạo CLS Token
        # self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        # nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 4. Transformer Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=4, 
            dim_feedforward=embed_dim*4, 
            batch_first=True, 
            norm_first=True, 
            dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Bước 1: Cắt dữ liệu đầu vào theo cấu trúc file JSON
        x_splits = torch.split(x, self.group_counts, dim=1)
        group_tokens = []
        
        # Bước 2: Nhúng và Lọc nhiễu (Attention Pooling)
        for i in range(len(self.group_counts)):
            # Nhúng các cột trong nhóm thành tokens
            toks = self.norms[i](self.tokenizers[i](x_splits[i]))
            
            # Tính trọng số Attention cho các cột trong nhóm
            weights = self.attn_pools[i](toks)
            
            # Ép về 1 token đại diện duy nhất (có chứa thông tin tinh hoa) cho nhóm đó
            pooled_token = (toks * weights).sum(dim=1, keepdim=True)
            group_tokens.append(pooled_token)
            
        # Bước 3: Ghép tất cả các token đại diện lại
        tokens = torch.cat(group_tokens, dim=1)
        
        # Bước 4: Chạy qua Transformer tiêu chuẩn
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x_trans = torch.cat([cls, tokens], dim=1)
        x_trans = self.transformer(x_trans)
        
        # Trả về dự đoán từ CLS token
        return self.head(x_trans[:, 0])