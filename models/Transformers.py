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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.trainer.callback_metrics]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in self.trainer.callback_metrics]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_acc', avg_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        return {'test_loss': avg_loss, 'test_acc': avg_acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def train(self, trainloader):
        self.trainer.fit(model, trainloader)
    
    def get_probability_for_true_class(self, x, y):
        prob_list = torch.softmax(self.forward(x), dim=1)
        return prob_list[y]



"""# MNIST Dataset loading and transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load MNIST
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# Model Instantiation
input_dim = 28  # Image size 28, for each row (token) in the transformer model
model = Transformer(input_dim=input_dim, d_model=256, n_layers=6, heads=8, n_mlp=4, n_classes=10)

# Training the model
model.train(train_loader)"""