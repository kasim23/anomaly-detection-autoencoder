from scipy.io import arff 
import pandas as pd 


def load_data(file_path):
    data = arff.loadarff(r'/home/hoover/u15/ssudais/ECG5000_TRAIN.arff') 
    train = pd.DataFrame(data[0]) 
    train["target"] = train['target'].str.decode("utf-8") 
    train.head() 