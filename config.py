import torch

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
TRAIN_DATA_PATH = r'/Users/omama/Documents/Portfolio/anomaly-detection/data/ECG5000_TRAIN.arff'
TEST_DATA_PATH = r'/Users/omama/Documents/Portfolio/anomaly-detection/data/ECG5000_TEST.arff'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")