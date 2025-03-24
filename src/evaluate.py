# src/evaluate.py
import torch
from models.autoencoder import RecurrentAutoencoder
from src.train import train_model

def evaluate_recall(model, dataset, threshold, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    for seq in dataset:
        seq = seq.to(device)
        rec = model(seq)
        error = criterion(rec, seq).item()
        # For normal sequences, we expect low reconstruction error.
        if error < threshold:
            correct += 1
    recall = correct / len(dataset)
    return recall

def experiment_thresholds(model, test_normal_dataset, test_anomaly_dataset, thresholds, criterion):
    recalls_normal = []
    recalls_anomaly = []
    for thresh in thresholds:
        recall_normal = evaluate_recall(model, test_normal_dataset, thresh, criterion)
        recall_anomaly = evaluate_recall(model, test_anomaly_dataset, thresh, criterion)
        recalls_normal.append(recall_normal)
        recalls_anomaly.append(recall_anomaly)
    return recalls_normal, recalls_anomaly


def experiment_embedding_dims(embedding_dims, train_dataset, val_dataset, test_normal_dataset, test_anomaly_dataset, seq_len, n_features, criterion, n_epochs=25):
    results = {}
    for emb_dim in embedding_dims:
        print(f"\nTraining with embedding dimension: {emb_dim}")
        model = RecurrentAutoencoder(seq_len, n_features, embedding_dim=emb_dim)
        trained_model, history = train_model(model, train_dataset, val_dataset, n_epochs)
        final_train_loss = history['train'][-1]
        final_val_loss = history['val'][-1]
        # Using a fixed threshold of 45 for evaluation.
        recall_normal = evaluate_recall(trained_model, test_normal_dataset, 45, criterion)
        recall_anomaly = evaluate_recall(trained_model, test_anomaly_dataset, 45, criterion)
        results[emb_dim] = {
            "train_loss": final_train_loss,
            "val_loss": final_val_loss,
            "recall_normal": recall_normal,
            "recall_anomaly": recall_anomaly
        }
    return results
