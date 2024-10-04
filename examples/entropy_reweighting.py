import os, glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torchcp.classification.predictors import SplitPredictor
from torchcp.classification.scores import APS, APSENTROPY, RAPS, SAPS

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_probs_and_labels(data_loader, classifier):
    probs_all = []
    ys_all = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = classifier(data)
            output = torch.softmax(output, dim=1)
            probs = output.cpu().detach().numpy()
            ys = target.cpu().detach().numpy()
            probs_all.append(probs)
            ys_all.append(ys)

    probs_all = np.concatenate(probs_all, axis=0)
    ys_all = np.concatenate(ys_all)

    return probs_all, ys_all

def temperature_scaling(cal_logits, cal_labels, val_logits, val_labels, score_function=APSENTROPY, alpha=0.01):
    results = []
    temperatures = np.logspace(-1, 1, 5)
    for temperature in temperatures:
        predictor = SplitPredictor(score_function(temperature))
        predictor.calculate_threshold(cal_logits, cal_labels, alpha) #calibrate(val_logits, val_labels, alpha)
        prediction_sets = predictor.predict(val_logits)

        coverage = predictor._metric('coverage_rate')(prediction_sets, val_labels)
        size = predictor._metric('average_size')(prediction_sets, val_labels)
        results.append(size)

    return temperatures[np.argmin(results)]

def evaluation(cal_logits, cal_labels, test_logits, test_labels, score_function, score_function_param=None, alpha=0.01):
    print(
        f"Experiment--Data : MNIST, Score : {score_function.__name__}, Predictor : SplitPredictor, Alpha : {alpha}")

    if score_function_param is not None:
        if isinstance(score_function_param, tuple):
            predictor = SplitPredictor(score_function(*score_function_param))
        else:
            predictor = SplitPredictor(score_function(score_function_param))
    else:
        predictor = SplitPredictor(score_function())

    predictor.calculate_threshold(cal_logits, cal_labels, alpha)  # calibrate(val_logits, val_labels, alpha)
    prediction_sets = predictor.predict(test_logits)

    coverage = predictor._metric('coverage_rate')(prediction_sets, test_labels)
    size = predictor._metric('average_size')(prediction_sets, test_labels)
    sscv = predictor._metric('SSCV')(prediction_sets, test_labels, alpha)

    print(f"Coverage: {np.round(coverage, 3)}; Size: {np.round(size, 3)}, SSCV: {np.round(sscv, 3)}")

    # Create a DataFrame to store the evaluation metrics
    metrics_df = pd.DataFrame({'Alpha': [alpha],
                               'Score Function': [score_function.__name__],
                               'Coverage': [coverage],
                               'Size': [size],
                               'SSCV': [sscv]})
    return metrics_df



# Initialize an empty DataFrame to store the evaluation metrics
results_df = pd.DataFrame(columns=['Dataset', 'Alpha', 'Score Function', 'Coverage', 'Size', 'SSCV'])

# Specify the folder path containing the logits and labels files
folder_path = 'data'

# Get a list of all logits and labels files in the folder
file_pattern = os.path.join(folder_path, 'logits_labels_*.pt')
files = glob.glob(file_pattern)

# Evaluate calibration for different sets of model_name, alpha, and score function
alphas = np.linspace(0.01, 0.10, 10)
score_functions = [APSENTROPY, APS, RAPS, SAPS]
score_function_params = {RAPS: (0.2, 2), SAPS: 0.2}
num_repetitions = 10

for file in files:
    try:
        data = torch.load(file)
    except:
        continue
    logits = data['logits']
    labels = data['labels']

    # Extract the dataset name from the file name
    dataset_name = os.path.basename(file).split('_')[2].split('.')[0]

    for alpha in alphas:
        for score_function in score_functions:
            coverage_values = []
            size_values = []
            sscv_values = []
            for run_id in range(num_repetitions):
                # Split the data into calibration and test sets
                cal_logits, test_logits, cal_labels, test_labels = train_test_split(logits, labels, test_size=0.2,
                                                                                    random_state=run_id)

                if score_function in score_function_params:
                    score_function_param = score_function_params[score_function]
                else:
                    score_function_param = None

                if score_function == APSENTROPY:
                    # Split the calibration set into validation and calibration sets
                    cal_logits, val_logits, cal_labels, val_labels = train_test_split(cal_logits, cal_labels,
                                                                                      test_size=0.5,
                                                                                      random_state=run_id)

                    score_function_param = temperature_scaling(cal_logits, cal_labels, val_logits, val_labels,
                                                               score_function, alpha)
                    print(f"Best temperature on calibration set is {score_function_param}")

                metrics_df = evaluation(cal_logits, cal_labels, test_logits, test_labels, score_function,
                                        score_function_param, alpha)
                coverage_values.append(metrics_df['Coverage'].values[0])
                size_values.append(metrics_df['Size'].values[0])
                sscv_values.append(metrics_df['SSCV'].values[0])

            # Calculate the median-of-means for coverage and size
            mean_coverage = pd.Series(coverage_values).mean()
            mean_size = pd.Series(size_values).mean()
            mean_sscv = pd.Series(sscv_values).mean()

            # Add the median-of-means results to the DataFrame with the dataset name
            results_df.loc[len(results_df)] = [dataset_name, alpha,
                                               f"Ours ({score_function})" if isinstance(score_function,
                                                                                        str) else score_function.__name__,
                                               mean_coverage, mean_size, mean_sscv]

results_df.to_csv('entropy.csv')