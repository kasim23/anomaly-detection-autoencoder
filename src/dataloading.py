from scipy.io import arff 
import pandas as pd 
from config import TRAIN_DATA_PATH, TEST_DATA_PATH

# def load_train_data():
#     data = arff.loadarff(TRAIN_DATA_PATH) 
#     train = pd.DataFrame(data[0]) 
#     train["target"] = train['target'].str.decode("utf-8")
#     # Optionally, printing the head for debugging but don't return it
#     print(train.head())
#     train_normal_df = train[train["target"] == "N"]
#     anomaly_df = train[train["target"] != "N"]
#     return train_normal_df, anomaly_df

def load_train_data():
    data = arff.loadarff(TRAIN_DATA_PATH) 
    train = pd.DataFrame(data[0]) 
    train["target"] = train['target'].str.decode("utf-8")
    # Optionally, print the head for debugging
    print(train.head())
    # Return the full training DataFrame
    return train
    
def load_test_data():
    data2 = arff.loadarff(TEST_DATA_PATH)
    test = pd.DataFrame(data2[0])
    test["target"] = test['target'].str.decode("utf-8")
    print("Test Data Head:")
    print(test.head())
    print("Unique target values:", test["target"].unique())
    
    # Adjust the condition based on your target values:
    # If normal heartbeats are labeled as 1 (or "1" after decoding), then use:
    test_normal_df = test[test["target"] == "1"]
    anomaly_df = test[test["target"] != "1"]
    
    return test_normal_df, anomaly_df
