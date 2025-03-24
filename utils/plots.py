import pandas as pd
import torch
import matplotlib.pyplot as plt

def plot_time_series_class(data, class_name, ax, n_steps=10):
    time_series_df = pd.DataFrame(data)

    # Smooth the time series using a rolling mean
    smooth_path = time_series_df.rolling(n_steps).mean()
    
    # Compute the deviation from the smooth path
    path_deviation = 2 * time_series_df.rolling(n_steps).std()

    # Calculate upper and lower lines for filling the area between them
    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]

    # Plot the smooth path
    ax.plot(smooth_path, linewidth=2)
    
    # Fill the area between the upper and lower lines
    ax.fill_between(
        path_deviation.index,
        under_line,
        over_line,
        alpha=.125
    )
    
    # Set the title of the plot to the class name
    ax.set_title(class_name)
    

def plot_threshold_results(thresholds, recalls_normal, recalls_anomaly):
    plt.figure()
    plt.plot(thresholds, recalls_normal, label="Normal Recall")
    plt.plot(thresholds, recalls_anomaly, label="Abnormal Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Recall")
    plt.legend()
    plt.title("Recall vs. Threshold")
    plt.show()
