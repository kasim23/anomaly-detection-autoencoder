import torch
import numpy as np
import pandas as pd


#Convert our examples into tensors, so we can use them to train the Autoencoder

def create_dataset(df):
  sequences = df.astype(np.float32).to_numpy().tolist()
  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
  n_seq, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features


#We'll combine the training and test data into a single data frame. This will give us more data to train our Autoencoder. We'll also shuffle it:
def combine_train_test(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([train, test], ignore_index=True)
    df = df.sample(frac=1.0)
    return df.shape