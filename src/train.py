import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
import pickle
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.transformer_model import TransformerModel
from src.data_preprocessing import load_data, preprocess_data, create_sequences
from src.utils import save_checkpoint


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(
            self.y[idx], dtype=torch.long
        )


def train():
    d_model = 256
    nhead = 8
    num_layers = 4
    dim_feedforward = 1024
    dropout = 0.1
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.0001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    df = load_data()
    char_to_idx, all_text = preprocess_data(df)
    X, y = create_sequences(all_text, char_to_idx)

    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}
    with open("char_to_idx.pkl", "wb") as f:
        pickle.dump(char_to_idx, f)
    with open("idx_to_char.pkl", "wb") as f:
        pickle.dump(idx_to_char, f)

    dataset = TextDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nStarting training...")
    print(f"Total batches per epoch: {len(dataloader)}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model.forward_full_sequence(inputs)
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(
            f"\nEpoch [{epoch+1}/{num_epochs}] completed, Average Loss: {avg_loss:.4f}\n"
        )

        if (epoch + 1) % 2 == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
                "vocab_size": vocab_size,
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "dropout": dropout,
            }
            save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
            print(f"Checkpoint saved at epoch {epoch+1}")

    checkpoint = {
        "epoch": num_epochs,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": avg_loss,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
    }
    save_checkpoint(checkpoint, filename="final_model.pth.tar")
    print("\nTraining completed! Final model saved.")


if __name__ == "__main__":
    train()
