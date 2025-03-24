# main.py
from src.train import train_model
from src.dataloading import load_train_data, load_test_data
from utils.datasets import create_dataset
import matplotlib.pyplot as plt
from models.autoencoder import RecurrentAutoencoder
from src.evaluate import experiment_thresholds, experiment_embedding_dims
import torch
from utils.plots import plot_threshold_results


if __name__ == "__main__":
    
    # Load your data (adjust these functions as needed)
    train_df = load_train_data()
    # Assume load_test_data returns a tuple: (test_normal_df, anomaly_df)
    test_normal_df, anomaly_df = load_test_data()
    
    # Create datasets:
    train_dataset, seq_len, n_features = create_dataset(train_df)
    # If you have a separate validation set, load/split it accordingly.
    # For simplicity, here we reuse the training dataset as validation.
    val_dataset = train_dataset  
    test_normal_dataset, _, _ = create_dataset(test_normal_df)
    test_anomaly_dataset, _, _ = create_dataset(anomaly_df)

    # Initialize the model with a default embedding dimension (8)
    model = RecurrentAutoencoder(seq_len, n_features, embedding_dim=8)

    # Train the model for 50 epochs
    model, history = train_model(model, train_dataset, val_dataset, n_epochs=50)

    # Plot training and validation loss
    plt.figure()
    plt.plot(history['train'], label="Train Loss")
    plt.plot(history['val'], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Over Training Epochs")
    plt.show()

    # --- Experiment 1: Varying the Threshold ---
    thresholds = list(range(15, 76, 10))
    criterion = torch.nn.L1Loss(reduction='sum')
    recalls_normal, recalls_anomaly = experiment_thresholds(model, test_normal_dataset, test_anomaly_dataset, thresholds, criterion)
    plot_threshold_results(thresholds, recalls_normal, recalls_anomaly)

    # --- Experiment 2: Varying the Embedding Dimension ---
    # embedding_dims = [2, 4, 6, 8]
    # emb_results = experiment_embedding_dims(embedding_dims, train_dataset, val_dataset, test_normal_dataset, test_anomaly_dataset, seq_len, n_features, criterion, n_epochs=25)
    # print("\nEmbedding Dimension Experiment Results:")
    # for emb_dim, metrics in emb_results.items():
    #     print(f"Embedding {emb_dim} -> Train Loss: {metrics['train_loss']:.2f}, Val Loss: {metrics['val_loss']:.2f}, "
    #           f"Normal Recall: {metrics['recall_normal']:.2f}, Abnormal Recall: {metrics['recall_anomaly']:.2f}")
