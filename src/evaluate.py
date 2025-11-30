import torch
import torch.nn as nn
import sys
import os
import pickle
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.transformer_model import TransformerModel
from src.data_preprocessing import load_data, preprocess_data, create_sequences
from src.train import TextDataset

def evaluate_model(model, dataloader, criterion, device, vocab_size):
   
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model.forward_full_sequence(inputs)
            
            outputs_flat = outputs.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)
            loss = criterion(outputs_flat, targets_flat)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs_flat, 1)
            total += targets_flat.size(0)
            correct += (predicted == targets_flat).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def calculate_perplexity(loss):

    return np.exp(loss)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    with open('char_to_idx.pkl', 'rb') as f:
        char_to_idx = pickle.load(f)
    with open('idx_to_char.pkl', 'rb') as f:
        idx_to_char = pickle.load(f)
    
    checkpoint = torch.load('final_model.pth.tar', map_location=device)
    vocab_size = checkpoint['vocab_size']
    d_model = checkpoint['d_model']
    nhead = checkpoint['nhead']
    num_layers = checkpoint['num_layers']
    dim_feedforward = checkpoint['dim_feedforward']
    dropout = checkpoint.get('dropout', 0.1)
    
    print(f"Model configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Attention heads: {nhead}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Feedforward dimension: {dim_feedforward}")
    print(f"  Training loss: {checkpoint['loss']:.4f}")
    
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    
    print("\nLoading evaluation data...")
    df = load_data()
    char_to_idx_data, all_text = preprocess_data(df)
    X, y = create_sequences(all_text, char_to_idx_data)
    
    eval_size = len(X) // 10
    X_eval = X[-eval_size:]
    y_eval = y[-eval_size:]
    
    eval_dataset = TextDataset(X_eval, y_eval)
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    criterion = nn.CrossEntropyLoss()
    print("\nEvaluating model...")
    avg_loss, accuracy = evaluate_model(model, eval_dataloader, criterion, device, vocab_size)
    perplexity = calculate_perplexity(avg_loss)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Perplexity: {perplexity:.2f}")
    print("="*50)