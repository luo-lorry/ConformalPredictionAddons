import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, ConcatDataset
from sklearn.preprocessing import StandardScaler

from torchcp.regression.predictors import SplitPredictor, CQR, CTI, R2CCP
from torchcp.regression.loss import QuantileLoss, R2ccpLoss
from torchcp.utils import fix_randomness
from torchcp.regression.utils import calculate_midpoints

from quantile_forest import RandomForestQuantileRegressor

from chr.methods import CHR
from chr.black_boxes import QNet, QRF
from chr.utils import evaluate_predictions


def save_results(results_df, filename):
    if os.path.exists(filename):
        existing_results_df = pd.read_csv(filename)
        results_df = pd.concat([existing_results_df, results_df], ignore_index=True)
    results_df.to_csv(filename, index=False)

def train(model, device, epoch, train_data_loader, criterion, optimizer):
    for index, (tmp_x, tmp_y) in enumerate(train_data_loader):
        outputs = model(tmp_x.to(device))
        loss = criterion(outputs, tmp_y.unsqueeze(dim=1).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def run_cti_rf(X_train, y_train, cal_data_loader, test_data_loader, alpha):
    quantiles = np.arange(0.01, 1.0, 0.01)
    model = RandomForestQuantileRegressor(min_samples_leaf=40)
    model.fit(X_train, y_train)

    predictor = CTI(model, quantiles, upper=y_train.max(), lower=y_train.min())
    predictor.calibrate(cal_data_loader, alpha)
    cti_results = predictor.evaluate(test_data_loader)
    return cti_results

def run_cti_nn(X_train, y_train, cal_data_loader, test_data_loader, alpha, epochs=2000, lr=0.0005, dropout=0.1):
    quantiles = np.arange(0.01, 1.0, 0.01)
    model = QNet(quantiles, num_features=X_train.shape[1], no_crossing=True, batch_size=X_train.shape[0],
                dropout=dropout, num_epochs=epochs, learning_rate=lr, calibrate=1, verbose=False)
    model.fit(X_train, y_train)

    predictor = CTI(model, quantiles, upper=y_train.max(), lower=y_train.min())
    predictor.calibrate(cal_data_loader, alpha)
    cti_results = predictor.evaluate(test_data_loader)
    return cti_results

def run_split_predictor(X_train, train_data_loader, cal_data_loader, test_data_loader, alpha, epochs=100):
    model = build_regression_model("NonLinearNet")(X_train.shape[1], 1, 64, 0.5).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        train(model, device, epoch, train_data_loader, criterion, optimizer)

    model.eval()
    predictor = SplitPredictor(model)
    predictor.calibrate(cal_data_loader, alpha)
    split_results = predictor.evaluate(test_data_loader)
    return split_results

def run_cqr(X_train, train_data_loader, cal_data_loader, test_data_loader, alpha, epochs=100):
    quantiles = [alpha / 2, 1 - alpha / 2]
    model = build_regression_model("NonLinearNet")(X_train.shape[1], 2, 64, 0.5).to(device)
    criterion = QuantileLoss(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        train(model, device, epoch, train_data_loader, criterion, optimizer)

    model.eval()
    predictor = CQR(model)
    predictor.calibrate(cal_data_loader, alpha)
    cqr_results = predictor.evaluate(test_data_loader)
    return cqr_results

def run_chr_nn(X_train, y_train, X_calib, y_calib, X_test, y_test, alpha, epochs=2000, lr=0.0005, dropout=0.1):
    grid_quantiles = np.arange(0.01, 1.0, 0.01)
    bbox = QNet(grid_quantiles, num_features=X_train.shape[1], no_crossing=True, batch_size=X_train.shape[0],
                dropout=dropout, num_epochs=epochs, learning_rate=lr, calibrate=1, verbose=False)
    bbox.fit(X_train, y_train)
    model = CHR(bbox, ymin=min(y_train), ymax=max(y_train), y_steps=1000, randomize=True)
    model.calibrate(X_calib, y_calib, alpha)
    pred = model.predict(X_test)
    res = evaluate_predictions(pred, y_test)
    return res

def run_chr_rf(X_train, y_train, X_calib, y_calib, X_test, y_test, alpha, n_estimators=100, min_samples_leaf=50):
    grid_quantiles = np.arange(0.01, 1.0, 0.01)
    bbox = QRF(grid_quantiles, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=2020, n_jobs=1, verbose=False)
    bbox.fit(X_train, y_train)
    model = CHR(bbox, ymin=min(y_train), ymax=max(y_train), y_steps=1000, randomize=True)
    model.calibrate(X_calib, y_calib, alpha)
    pred = model.predict(X_test)
    res = evaluate_predictions(pred, y_test)
    return res

def run_r2ccp(train_dataset, cal_dataset, train_data_loader, cal_data_loader, test_data_loader, alpha, epochs=100):
    p = 0.5
    tau = 0.2
    K = 50
    train_and_cal_dataset = ConcatDataset([train_dataset, cal_dataset])
    train_and_cal_data_loader = torch.utils.data.DataLoader(train_and_cal_dataset, batch_size=100, shuffle=True, pin_memory=True)
    midpoints = calculate_midpoints(train_and_cal_data_loader, K)

    model = build_regression_model("NonLinearNet")(train_dataset[0][0].shape[0], K, 1000, 0).to(device)
    criterion = R2ccpLoss(p, tau, midpoints)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(epochs):
        train(model, device, epoch, train_data_loader, criterion, optimizer)

    model.eval()
    predictor = R2CCP(model, midpoints)
    predictor.calibrate(cal_data_loader, alpha)
    r2ccp_results = predictor.evaluate(test_data_loader)
    return r2ccp_results


def run_experiments(data_names, num_runs, ratio_train_values, test_ratio, alpha_values, results_file):
    results = []
    methods = ['CTI(RF)', 'CTI(NN)', 'Split', 'CQR', 'CHR(NN)', 'CHR(RF)']

    if os.path.exists(results_file):
        existing_results_df = pd.read_csv(results_file)
    else:
        existing_results_df = pd.DataFrame(columns=['Experiment_Key', 'Dataset', 'Ratio_Train', 'Alpha'] +
                                                   [f'{method}_{metric}' for method in methods for metric in ['Coverage_Rate', 'Average_Size']])

    for data_name in data_names:
        for ratio_train in ratio_train_values:
            for alpha in alpha_values:
                method_results = {method: {'coverage': [], 'size': []} for method in methods}

                for run in range(num_runs):
                    experiment_key = f"{data_name}_{ratio_train}_{alpha}"
                    if experiment_key in existing_results_df['Experiment_Key'].values:
                        print(f"Skipping completed experiment: {experiment_key}")
                        continue
                    print(f"Running dataset: {data_name}, Ratio_Train: {ratio_train}, Alpha: {alpha}, Run: {run + 1}/{num_runs}")
                    seed = run + 1
                    X_train, X_calib, X_test, y_train, y_calib, y_test = build_reg_data(data_name, ratio_train, test_ratio, seed)

                    # Prepare datasets and data loaders
                    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
                    cal_dataset = TensorDataset(torch.from_numpy(X_calib), torch.from_numpy(y_calib))
                    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

                    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True)
                    cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=100, shuffle=False, pin_memory=True)
                    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=True)

                    # CTI + RF
                    cti_results = run_cti_rf(X_train, y_train, cal_data_loader, test_data_loader, alpha)
                    method_results['CTI(RF)']['coverage'].append(cti_results['Coverage_rate'])
                    method_results['CTI(RF)']['size'].append(cti_results['Average_size'])

                    # CTI + NN
                    cti_results = run_cti_nn(X_train, y_train, cal_data_loader, test_data_loader, alpha)
                    method_results['CTI(NN)']['coverage'].append(cti_results['Coverage_rate'])
                    method_results['CTI(NN)']['size'].append(cti_results['Average_size'])

                    # SplitPredictor
                    split_results = run_split_predictor(X_train, train_data_loader, cal_data_loader, test_data_loader, alpha)
                    method_results['Split']['coverage'].append(split_results['Coverage_rate'].item())
                    method_results['Split']['size'].append(split_results['Average_size'].item())

                    # CQR
                    cqr_results = run_cqr(X_train, train_data_loader, cal_data_loader, test_data_loader, alpha)
                    method_results['CQR']['coverage'].append(cqr_results['Coverage_rate'].item())
                    method_results['CQR']['size'].append(cqr_results['Average_size'].item())

                    # CHR + NN
                    chr_nn_results = run_chr_nn(X_train, y_train, X_calib, y_calib, X_test, y_test, alpha)
                    method_results['CHR(NN)']['coverage'].append(chr_nn_results['Coverage'])
                    method_results['CHR(NN)']['size'].append(chr_nn_results['Length'])

                    # CHR + RF
                    chr_rf_results = run_chr_rf(X_train, y_train, X_calib, y_calib, X_test, y_test, alpha)
                    method_results['CHR(RF)']['coverage'].append(chr_rf_results['Coverage'])
                    method_results['CHR(RF)']['size'].append(chr_rf_results['Length'])


                result = {
                    'Experiment_Key': experiment_key,
                    'Dataset': data_name,
                    'Ratio_Train': ratio_train,
                    'Alpha': alpha,
                    **{
                        f'{method}_{metric}_{stat}': value
                        for method in methods
                        for metric, stat, value in [
                            ('Coverage_Rate', 'Avg', np.mean(method_results[method]['coverage'])),
                            ('Coverage_Rate', 'Std', np.std(method_results[method]['coverage'])),
                            ('Average_Size', 'Avg', np.mean(method_results[method]['size'])),
                            ('Average_Size', 'Std', np.std(method_results[method]['size']))
                        ]
                    }
                }
                results.append(result)
                df = pd.DataFrame(results)
                save_results(df, results_file)

    return pd.concat([existing_results_df, df], ignore_index=True)


if __name__ == '__main__':
    from datetime import datetime
    results_file = f'experiment_results_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.csv'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_names = ['bike', 'blog_data', 'community', 'star', 'homes', 'meps_19', 'meps_20', 'meps_21', 'facebook_1', 'facebook_2', 'bio', 'concrete']
    num_runs = 10
    ratio_train_values = [0.7]
    test_ratio = 0.2
    alpha_values = [0.1]

    results_df = run_experiments(data_names, num_runs, ratio_train_values, test_ratio, alpha_values, results_file)
    print(results_df)
