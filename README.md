# Heartbeat Anomaly Detection with LSTM Autoencoders

This repository contains a project aimed at detecting abnormal heartbeats (anomalies) using a Recurrent Autoencoder (LSTM-based). The dataset consists of ECG signals, each representing a single heartbeat with 140 time steps.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Data Description](#data-description)  
3. [Objectives](#objectives)  
4. [Project Structure](#project-structure)  
5. [Method & Architecture](#method--architecture)  
6. [Setup Instructions](#setup-instructions)  
7. [Usage](#usage)  
8. [Results So Far](#results-so-far)  
9. [What You Can Learn](#what-you-can-learn)  
10. [Future Work](#future-work)

---

## Project Overview
This project focuses on **anomaly detection** in ECG (electrocardiogram) signals. We use a **Recurrent Autoencoder** to learn the “normal” heartbeat pattern. Any heartbeat that deviates significantly (based on reconstruction error) is flagged as an anomaly.  

### Why Autoencoders for Anomaly Detection?
- Autoencoders are trained to reconstruct normal data.  
- If an input is significantly different from the training distribution, the reconstruction error will be higher, indicating an anomaly.

---

## Data Description
We use a dataset with 5,000 heartbeat time-series, each with 140 time steps. The classes are labeled from 1 to 5, where:
- **1** = Normal  
- **2, 3, 4, 5** = Different types of abnormal heartbeats  

For simplicity, the dataset is split into:
- **Train Data**: Used to train the autoencoder, primarily containing normal heartbeats.  
- **Test Data**: Contains both normal and abnormal heartbeats to evaluate how well the model detects anomalies.

The data is stored in **ARFF** files, which are loaded using `scipy.io.arff`.

---

## Objectives
1. **Train an LSTM-based autoencoder** to learn normal heartbeat patterns.
2. **Determine a reconstruction error threshold** to separate normal vs. abnormal heartbeats.
3. **Evaluate performance** by measuring recall (and potentially other metrics) for both normal and abnormal classes across varying thresholds.
4. **Explore hyperparameter tuning** (e.g., embedding dimension, learning rate, etc.) to improve detection.
5. **Containerize and deploy** the final model for real-world usage (future milestone).
6. **Integrate MLflow** for experiment tracking (future milestone).

---

## Project Structure

```
anomaly-detection/
├── main.py
├── config.py
├── requirements.txt       (optional: if you list your dependencies)
├── results/               (stores model checkpoints, plots, etc.)
├── src/
│   ├── __init__.py
│   ├── train.py           (training loop and logic)
│   ├── evaluate.py        (evaluation utilities for thresholds, recall, etc.)
│   └── dataloading.py     (data loading functions)
├── models/
│   ├── __init__.py
│   ├── autoencoder.py     (RecurrentAutoencoder class)
│   ├── encoder.py         (Encoder class with LSTM layers)
│   └── decoder.py         (Decoder class with LSTM layers)
└── utils/
    ├── __init__.py
    ├── datasets.py        (dataset creation from dataframes)
    └── plots.py           (plotting utilities for time series, thresholds, etc.)
```

- **`main.py`**: Orchestrates data loading, model instantiation, training, and evaluation.  
- **`config.py`**: Centralizes configuration, including device (CPU/GPU) and file paths.  
- **`results/`**: Contains saved plots, model checkpoints, and other outputs.  
- **`src/`**: Houses core training and evaluation scripts, plus data loading utilities.  
- **`models/`**: Defines the LSTM-based autoencoder architecture.  
- **`utils/`**: Contains helper functions for dataset creation and plotting.

---

## Method & Architecture

### Recurrent Autoencoder
1. **Encoder (LSTM)**  
   - Takes the time-series input (shape: 140 x 1).  
   - Compresses it into a latent representation of size `embedding_dim`.

2. **Decoder (LSTM)**  
   - Takes the latent representation and attempts to reconstruct the original time series.  
   - Reconstruction error is measured with an L1 or MSE loss.

**Why LSTM?**  
- ECG signals are sequential data.  
- LSTM can capture temporal dependencies better than simple feedforward networks.

---

## Setup Instructions

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/anomaly-detection.git
   cd anomaly-detection
   ```

2. **Create and Activate a Virtual Environment** (optional but recommended)  
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or on Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   If you don’t have a `requirements.txt`, install packages manually:
   ```bash
   pip install torch pandas numpy scipy matplotlib
   ```

4. **Configure Paths**  
   - In `config.py`, update `TRAIN_DATA_PATH` and `TEST_DATA_PATH` to point to your `.arff` files.  
   - Make sure `DEVICE` is set to use GPU if available, otherwise CPU.

---

## Usage

1. **Run the Main Script**  
   From the project root:
   ```bash
   python main.py
   ```
   - This loads the training and test data, trains the LSTM autoencoder, and plots the training/validation loss curves.
   - It also evaluates recall for normal vs. abnormal samples across varying thresholds.

2. **Check the `results/` Folder**  
   - You’ll find plots (e.g., `loss_over_epochs.png`, `recall_vs_threshold.png`) and a saved model checkpoint (e.g., `best_model.pt`).

---

## Results So Far

- **Training Loss** steadily decreases, indicating the autoencoder is learning to reconstruct normal heartbeats.  
- **Validation Loss** also decreases, though typically not as low as the training loss, showing some generalization.  
- **Recall vs. Threshold** shows how different threshold values affect detection of normal vs. abnormal heartbeats.  


---

## What You Can Learn

By exploring this repository, you will gain hands-on experience in:

1. **Time Series Processing**  
   - Loading and preprocessing ECG data from ARFF files.  
   - Reshaping data for PyTorch training.

2. **LSTM Autoencoders**  
   - Constructing and training recurrent networks for reconstruction tasks.  
   - Understanding how reconstruction error can be used for anomaly detection.

3. **Hyperparameter Tuning**  
   - Experimenting with embedding dimensions, LSTM hidden sizes, and thresholds.  
   - Observing how these affect training/validation loss and anomaly detection metrics.

4. **Data Visualization**  
   - Plotting training/validation loss curves.  
   - Plotting recall vs. threshold for anomaly detection.

5. **Future Extensions** (planned)  
   - **MLflow Integration** for experiment tracking, parameter logging, and artifact storage.  
   - **Containerization & Deployment** with Docker, enabling easy model serving and scaling.  
   - **Cloud Deployment** (AWS, GCP, or similar) for production-ready inference.

---

## Future Work

We will be extending this project to include:

1. **MLflow Experiment Tracking**  
   - Track multiple runs with different hyperparameters.  
   - Automatically store plots, metrics, and model checkpoints.

2. **Containerization & Deployment**  
   - Build a Docker image that packages the trained model and inference code.  
   - Deploy the model as a microservice on a platform like AWS ECS/EKS or another orchestration tool.

3. **Real-Time or Batch Inference**  
   - Provide a REST API or batch processing script for new ECG signals.  
   - Return anomaly scores or binary classification (normal vs. abnormal).

Stay tuned for updates as we integrate these new features!

---

_We hope this README helps you get started with the project. If you have any questions, feel free to open an issue or reach out._