import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Inspired from Deep Learning Problem Sheet 6

class SelfAttention(nn.Module):
    def __init__(self, d: int, heads: int = 8):
        super().__init__()
        self.k, self.h = d, heads
        self.Wq = nn.Linear(d, d * heads, bias=False)
        self.Wk = nn.Linear(d, d * heads, bias=False)
        self.Wv = nn.Linear(d, d * heads, bias=False)
        self.unifyheads = nn.Linear(heads * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.size()
        h = self.h

        queries = self.Wq(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b * h, l, d)
        keys = self.Wk(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b * h, l, d)
        values = self.Wv(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b * h, l, d)

        w_prime = torch.bmm(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(d, dtype=torch.float32))
        w = F.softmax(w_prime, dim=-1)
        out = torch.bmm(w, values).view(b, h, l, d)
        out = out.transpose(1, 2).contiguous().view(b, l, h * d)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, d: int, heads: int = 8, n_mlp: int = 4):
        super().__init__()
        self.attention = SelfAttention(d, heads=heads)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, n_mlp * d),
            nn.ReLU(),
            nn.Linear(n_mlp * d, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_prime = self.attention(x)
        x = self.norm1(x_prime + x)
        x_prime = self.ff(x)
        return self.norm2(x_prime + x)


class Transformer(pl.LightningModule):
    def __init__(self, grayscale=True, patch_size: int = 7, d_model: int = 256, n_layers: int = 10, heads: int = 8, n_mlp: int = 4, n_classes: int = 10):
        super().__init__()
        self.d_model = d_model
        # For Vision Transformers, divide image into patches
        self.patch_size = patch_size
        if grayscale:
            self.patch_embedding = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
        else:
            self.patch_embedding = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
 

        self.blocks = nn.ModuleList([TransformerBlock(d_model, heads, n_mlp) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, n_classes)

    def forward(self, x):
        b = x.size(0)  # Batch size
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1,2)
        
        # Pass through each transformer block
        for block in self.blocks:
            x = block(x)
        
        logits = self.fc_out(x.mean(dim=1))  # Shape: [batch_size, n_classes]
        return logits
