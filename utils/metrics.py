import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe



import numpy as np
import pandas as pd
class Overall_Metrics:
    def __init__(self):
        self.mae = 0  # 各个变量的 MAE
        self.mse = 0  # 各个变量的 MSE
        self.rmse = 0
        self.mape = 0
        self.mspe = 0
        self.count = 0

    def update(self, predicted_values, true_values):
        self.mae += MAE(predicted_values, true_values)
        self.mse += MSE(predicted_values, true_values)
        self.mape += MAPE(predicted_values, true_values)
        self.mspe += MSPE(predicted_values, true_values)

        self.count += 1

    def get_metrics(self):
        mae = self.mae / self.count
        mse = self.mse / self.count
        rmse = np.sqrt(self.mse / self.count)
        mape = self.mape / self.count
        mspe = self.mspe / self.count

        return mae, mse, rmse, mape, mspe

def SEDI(predicted_values, true_values, percentile):
    percentile = percentile.numpy()
    num_percentile = percentile.shape[-1]
    weight = np.array([ [0.5,0.5],
                        [0.5,0.5],
                        [0.5,0.5], # ignore wind direction actually 
                        [0,   1],  # wind speed 
                        [0.5,0.5]
                      ])
    gt_events_list = []
    pred_events_list = []

    for i in range(num_percentile // 2):
        gt_events_low = (true_values < percentile[:,:,:,i])
        pred_events_low = np.sum(np.logical_and(predicted_values< percentile[:,:,:,i], gt_events_low),axis=(0,1)) 
        
        gt_events_high = (true_values > percentile[:,:,:, num_percentile-1-i])
        pred_events_high = np.sum(np.logical_and(predicted_values > percentile[:,:,:,num_percentile-1-i], gt_events_high),axis=(0,1))
        
        gt_events = np.sum(gt_events_low,axis=(0,1))+ np.sum(gt_events_high,axis=(0,1))

        gt_events_list.append(gt_events)
        pred_events_list.append(pred_events_high + pred_events_low)

    return np.array(pred_events_list), np.array(gt_events_list)


class MultiMetricsCalculator:
    def __init__(self):
        self.mae = np.zeros(5)  # 各个变量的 MAE
        self.mse = np.zeros(5)  # 各个变量的 MSE
        self.SEDI_pred = np.zeros((4,5))
        self.SEDI_gt = np.zeros((4,5))
        self.count = 0

    def update(self, predicted_values, true_values, percentile=None):
        absolute_errors = np.abs(predicted_values - true_values)
        squared_errors = np.square(predicted_values - true_values)

        self.mae += np.mean(absolute_errors, axis=(0, 1))
        self.mse += np.mean(squared_errors, axis=(0, 1))

        pred_events, gt_events = SEDI(predicted_values, true_values, percentile)
        self.SEDI_pred +=pred_events
        self.SEDI_gt += gt_events
        # print(self.SEDI_pred/self.SEDI_gt)
        # import pdb
        # pdb.set_trace()
        self.count += 1

    def get_metrics(self):
        avg_mae = self.mae / self.count
        avg_mse = self.mse / self.count
        SEDI = self.SEDI_pred / self.SEDI_gt
        return avg_mae, avg_mse, SEDI

# 创建一个 WeatherMetricsCalculator 对象
calculator = MultiMetricsCalculator()

