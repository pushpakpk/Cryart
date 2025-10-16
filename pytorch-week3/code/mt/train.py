import torch
import torch.nn as nn
import torch.optim as optim
from mt.dataset import get_toy_loaders
from mt.model import Transformer
import os
from tqdm import tqdm
from utils.viz import plot_curves

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_masks(src, tgt, pad_idx=0):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    seq_len = tgt.size(1)
    # causal mask
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=src.device)).bool()
    tgt_mask = tgt_mask & causal_mask
    return src_mask, tgt_mask

def train_model(epochs=20, batch_size=32, save_dir='runs/mt/'):
    os.makedirs(save_dir, exist_ok=True)
    loader, src_vocab, tgt_vocab = get_toy_loaders(batch_size)
    model = Transformer(len(src_vocab), len(tgt_vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for src, tgt, _, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            tgt_input = tgt[:,:-1]
            tgt_labels = tgt[:,1:]
            src_mask, tgt_mask = create_masks(src, tgt_input)
            outputs = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_labels.reshape(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(save_dir,'transformer_best.pth'))

    plot_curves(train_losses, save_dir)

if __name__ == "__main__":
    train_model()
