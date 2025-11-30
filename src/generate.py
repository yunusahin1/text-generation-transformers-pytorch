import torch
import sys
import os
import pickle
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.transformer_model import TransformerModel


def generate_text(
    model,
    start_text,
    char_to_idx,
    idx_to_char,
    length=500,
    temperature=1.0,
    device="cpu",
):

    model.eval()

    start_text = start_text.lower()

    current_seq = [char_to_idx.get(ch, 0) for ch in start_text[-100:]]
    generated_text = start_text

    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([current_seq], dtype=torch.long).to(device)

            output = model(x)

            output = output / temperature
            probs = torch.softmax(output, dim=1)

            probs_np = probs.cpu().numpy().flatten()
            next_idx = np.random.choice(len(probs_np), p=probs_np)

            next_char = idx_to_char[next_idx]
            generated_text += next_char

            current_seq = current_seq[1:] + [next_idx]
            if len(current_seq) > 100:
                current_seq = current_seq[-100:]

    return generated_text


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("char_to_idx.pkl", "rb") as f:
        char_to_idx = pickle.load(f)
    with open("idx_to_char.pkl", "rb") as f:
        idx_to_char = pickle.load(f)

    checkpoint = torch.load("final_model.pth.tar", map_location=device)
    vocab_size = checkpoint["vocab_size"]
    d_model = checkpoint["d_model"]
    nhead = checkpoint["nhead"]
    num_layers = checkpoint["num_layers"]
    dim_feedforward = checkpoint["dim_feedforward"]
    dropout = checkpoint.get("dropout", 0.1)

    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    start_text = "in this paper we present a novel approach to"
    print("Starting text:", start_text)
    print("\nGenerating text...\n")
    print("=" * 80)

    generated = generate_text(
        model,
        start_text,
        char_to_idx,
        idx_to_char,
        length=500,
        temperature=0.8,
        device=device,
    )

    print(generated)
    print("=" * 80)
