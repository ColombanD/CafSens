import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# From Deep Learning Problem Sheet 6

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
    def __init__(self, input_dim: int, d_model: int = 256, n_layers: int = 6, heads: int = 8, n_mlp: int = 4, n_classes: int = 10):
        super().__init__()
        self.d_model = d_model
        self.token_projection = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, heads, n_mlp) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, n_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

        self.trainer = pl.Trainer(max_epochs=5)

    def forward(self, x):
        b = x.size(0)  # Batch size
        x = x.view(b, 28, 28)  # Shape: [batch_size, seq_len=28, input_dim=28] (each row is a token)
        x = self.token_projection(x)  # Shape: [batch_size, seq_len=28, d_model]
        
        # Pass through each transformer block
        for block in self.blocks:
            x = block(x)
        
        logits = self.fc_out(x.mean(dim=1))  # Shape: [batch_size, n_classes]
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def train(self, trainloader):
        self.trainer.fit(self, trainloader)
    
    def get_probability_for_true_class(self, x, y):
        # Softmax the output of the model to get the probabilities
        prob_list = torch.softmax(self.forward(x), dim=1)
        prob_for_true_class = []
        # Since we have a batch of inputs, we need to get the probability of the true class for each input of the batch
        for i in range(len(prob_list)):
            prob_for_true_class.append(prob_list[i][y[i]])
        return prob_for_true_class
