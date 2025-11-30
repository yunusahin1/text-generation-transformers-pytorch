import torch
import os
from datasets import load_dataset
import pandas as pd
from typing import List, Dict


def load_data() -> pd.DataFrame:
    if not os.path.exists("data/abstracts.csv"):
        os.makedirs("data", exist_ok=True)
        dataset = load_dataset("nick007x/arxiv-papers", split="train")
        dataset.to_csv("data/abstracts.csv", index=False)
    df = pd.read_csv("data/abstracts.csv")
    return df


def preprocess_data(df: pd.DataFrame) -> Dict[str, int]:
    df = df.dropna(subset=["abstract"])[["abstract"]]
    df = df.rename(columns={"abstract": "text"})[:50]
    df["text"] = df["text"].str.lower()
    texts = df["text"].tolist()
    all_text = "".join(texts)
    # remove redundant characters
    all_text = all_text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    chars = sorted(list(set(all_text)))
    print(f"Total characters: {len(all_text)}")
    print(f"Unique characters: {len(chars)}")
    char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
    return char_to_idx, all_text


def create_sequences(
    all_text: str, char_to_idx: Dict[str, int]
) -> (List[List[int]], List[List[int]]):

    seq_length = 100
    step_size = 1
    X = []
    y = []
    for i in range(0, len(all_text) - seq_length, step_size):
        seq = all_text[i : i + seq_length]
        target_seq = all_text[i + 1 : i + seq_length + 1]
        X.append([char_to_idx[ch] for ch in seq])
        y.append([char_to_idx[ch] for ch in target_seq])
    print(f"Total sequences: {len(X)}")
    return X, y
