# utils/save_embeddings.py
import numpy as np
import pickle


def save_embeddings(embeddings, chunks, path="embeddings_data.pkl"):
    with open(path, "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "chunks": chunks
        }, f)


def load_embeddings(path="embeddings_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data["embeddings"], data["chunks"]