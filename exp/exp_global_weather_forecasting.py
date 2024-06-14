from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual,visual_multiple
from utils.metrics import Overall_Metrics, MultiMetricsCalculator
from utils.logger import Logger
import torch
import torch.nn as nn
from torch import optim
import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import cycle
warnings.filterwarnings('ignore')

class Exp_Global_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Global_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader),\
                desc="Calculating metrics", total=len(vali_loader), unit="site"):
                batch_x = batch_x.float().to(self.device).squeeze(0)
                batch_y = batch_y.float().squeeze(0)

                batch_x_mark = batch_x_mark.float().to(self.device).squeeze(0)
                batch_y_mark = batch_y_mark.float().to(self.device).squeeze(0)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
     
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                # print(loss)
                total_loss.append(loss)
                if (i+1)==len(vali_loader):
                    break
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        os.makedirs(f'{self.args.checkpoints}/{setting}', exist_ok=True)

        log = Logger(f'{self.args.checkpoints}/{setting}/train.log')

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = self.args.train_steps#len(train_loader)
        train_loader = iter(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        # vali_loss = self.vali(vali_data, vali_loader, criterion)

        epoch_time = time.time()
        train_loss = []
        for iter_count in range(train_steps):
            self.model.train()
 
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(train_loader)
         
            model_optim.zero_grad()
            batch_x = batch_x.float().to(self.device).squeeze(0)

            batch_y = batch_y.float().to(self.device).squeeze(0)
            batch_x_mark = batch_x_mark.float().to(self.device).squeeze(0)
            batch_y_mark = batch_y_mark.float().to(self.device).squeeze(0)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
            
            lr = adjust_learning_rate(model_optim, iter_count + 1, self.args)

            print_freq=100
            if (iter_count + 1) % print_freq == 0:
                speed = (time.time() - time_now) / print_freq
                left_time = speed * (train_steps - iter_count)
                minutes, seconds = divmod(int(left_time), 60)
                hours, minutes = divmod(minutes, 60)
                days, hours = divmod(hours, 24)

                log.info("iters: {0}/{1}, epoch: {2} |lr:{3:.7f} loss: {4:.5f} speed: {5:.2f}s/it eta: {6:02d}:{7:02d}:{8:02d}".format(
                    iter_count+1,train_steps, 1, lr, loss.item(), speed, days, hours, minutes))
                time_now = time.time()

            if torch.isnan(loss).any():
                import pdb
                pdb.set_trace()

            if self.args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()


            if (iter_count+1) % self.args.val_steps == 0:
                log.info("Epoch: {} cost time: {}".format( 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                # test_loss = self.vali(test_data, test_loader, criterion)

                log.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    1, train_steps, train_loss, vali_loss, vali_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    log.info("Early stopping")
                    break
                train_loss = []
               
            if iter_count%2000 ==0:
                torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
        best_model_path = path + '/' + 'checkpoint.pth'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(best_model_path, map_location=device))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'),map_location=device))

        preds = []
        trues = []
        vis_folder_path = f'{self.args.checkpoints}/{setting}/vis'
        os.makedirs(vis_folder_path, exist_ok=True)
        metric_multi = MultiMetricsCalculator()
        metric_overall = Overall_Metrics()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, percentile) in tqdm(enumerate(test_loader), \
                desc="Calculating metrics", total=len(test_loader), unit="site"):

                batch_x = batch_x.float().to(self.device).squeeze(0)
                batch_y = batch_y.float().to(self.device).squeeze(0)

                batch_x_mark = batch_x_mark.float().to(self.device).squeeze(0)
                batch_y_mark = batch_y_mark.float().to(self.device).squeeze(0)
                percentile = percentile.squeeze(0)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y).reshape(shape)
                # import pdb
                # pdb.set_trace()
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                # preds.append(pred)
                # trues.append(true)
                metric_multi.update(pred, true, percentile)
                metric_overall.update(pred, true)
                # for visualization
                if i % 50 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input).reshape(shape)
                    gt = np.concatenate((input[0, :, :], true[0, :, :]), axis=0)
                    pred = np.concatenate((input[0, :, :], pred[0, :, :]), axis=0)
                    visual_multiple(gt, pred, os.path.join(vis_folder_path, str(i) + '.pdf'),
                    ['Temperature', 'Dewpoint', 'Wind Angle', 'Wind Rate', 'Sea-level Pressure'])
                if (i+1) == len(test_loader):
                    break
        avg_mae, avg_mse, SEDI = metric_multi.get_metrics()

        print(avg_mae, avg_mse)

        # preds = np.concatenate(preds, 0)
        # trues = np.concatenate(trues, 0)
        
        mae, mse, rmse, mape, mspe = metric_overall.get_metrics()


        # creat a dataFrame to save mutli-metrics
        metrics_df = pd.DataFrame({'Variable': ['Temperature', 'Dewpoint Temperature', 'Wind Angle', 'Wind Rate','Sea-level Pressure'],
                                'MAE': avg_mae.tolist(),
                                'MSE': avg_mse.tolist(),
                                'SEDI_99.5': SEDI[0,:].tolist(),
                                'SEDI_98': SEDI[1,:].tolist(),
                                'SEDI_95': SEDI[2,:].tolist(),
                                'SEDI_90': SEDI[3,:].tolist(),
                                'all': [mae, mse, rmse, mape, mspe]})  
        print(metrics_df)
        # print('test shape:', preds.shape, trues.shape)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)
        print('mse:{}, mae:{}'.format(mse, mae))
        # result save
        metric_folder_path = f'{self.args.checkpoints}/{setting}/'
        
        os.makedirs(metric_folder_path, exist_ok=True)

        metrics_df.to_csv(f'{metric_folder_path}metric.csv', index=False)

        # f = open("result_long_term_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return
