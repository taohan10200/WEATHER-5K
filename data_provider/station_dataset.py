import os
import numpy as np
import pandas as pd
import glob
import re
import torch
import copy
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from multiprocessing import Manager
from multiprocessing import shared_memory,Pool,Manager,Lock
import multiprocessing as mp
import concurrent.futures
import threading
import queue
import json
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from utils.tools import StandardScaler
warnings.filterwarnings('ignore')
class Dataset_Weather_Stations(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path=None,
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag =flag
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path

        # with open(f'{self.root_path}/meta_info.json') as f:
        #     station_info = json.load(f)
        # self.station_names = [f'{root_path}/raw_inter_era5/{i}' for i in list(station_info.keys())]
        self.station_names = glob.glob(f'{self.root_path}/raw_inter_era5/*.csv')
        self.num_station = len(self.station_names)
        if  flag=='train':
            self.timestamp = pd.date_range(start='2014-01-01', end='2021-12-31-23', freq='1H')
        elif  flag=='val':
            self.timestamp = pd.date_range(start='2022-01-01', end='2022-12-31-23', freq='1H')
        elif  flag=='test':
            with open(f'{self.root_path}/percentile.json') as f:
                dict_data = json.load(f)
            self.percentiles = []
            for station_name in self.station_names:
                station_name = station_name.split('/')[-1] 

                self.percentiles.append(np.array(dict_data[station_name])[None])
            self.percentiles = np.concatenate(self.percentiles, axis=0)

            self.timestamp = pd.date_range(start='2023-01-01', end='2023-12-31-23', freq='1H')
        self.num_timestamp_input = len(self.timestamp) - (self.seq_len + self.pred_len - 1)

        print(f'{self.flag} samples: {len(self.timestamp)}')
        
        data_shape = (self.num_station, len(self.timestamp), 9)
        shared_data = np.zeros(data_shape, dtype=np.float32)
 
        self.shm = shared_memory.SharedMemory( create=True, size=shared_data.nbytes)
        del shared_data
        self.shared_data_np = np.ndarray(data_shape, dtype=np.float32, buffer=self.shm.buf)
        self.lock = Lock()
        self.data_stamp = None

        self.scaler = StandardScaler(
            mean=np.array([12.70852261, 6.52705582,191.18867587,3.36941836,1014.85317029])[None,:], 
            std=np.array([13.08167293, 12.13875438, 99.67403125,  2.65729403,  9.17480999])[None,:])

        self.station_Index2Name = {index: name for name, index in enumerate(self.station_names)}
        self.registed_StationIndex = Manager().list()
        self.registed_completed = False
 
        self.stop_event = threading.Event()
        self.pre_load()

    def pre_load(self):
        # create a daemon to pre-load station data
        def check_progress():
            for i in range(len(self.station_names)):
                self.registed_shared_data(i)
                if self.registed_completed:
                    self.stop_event.set()
        progress_thread = threading.Thread(target=check_progress)
        progress_thread.daemon = True  # set as a daemon
        progress_thread.start()

    def registed_shared_data(self, station_index):
        station_name = self.station_names[station_index]
        station_index = self.station_Index2Name.get(station_name)
        df_raw = pd.read_csv(station_name)
    
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        # cols.remove('DATE')
        self.target = ['DATE','TMP','DEW', 'WND_ANGLE', 'WND_RATE', 'SLP']
        # df_raw = df_raw[['DATE'] + cols + [self.target]]
        df_raw = df_raw[self.target ]
        # import pdb
        # pdb.set_trace()
        border_st = df_raw[df_raw.iloc[:, 0] == str(self.timestamp[0])].index[0] # get the start
        border_ed = border_st + len(self.timestamp)
        df_raw=df_raw[border_st:border_ed]
        # print(f'phase: {self.flag}, from {border_st} to {border_ed}')

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['DATE']]
        df_stamp['DATE'] = pd.to_datetime(df_stamp.DATE)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['DATE'], 1).values

        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['DATE'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        tmp_data = np.concatenate([data_stamp, data], axis=1, dtype=np.float32)
        
        self.lock.acquire()
        if station_index not in self.registed_StationIndex: 
            self.shared_data_np[station_index, :, :] = tmp_data 
            self.registed_StationIndex.append(station_index)
        self.lock.release()

    def multi_process_load(self, station_indexs):
        num_threads = mp.cpu_count() 
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(self.registed_shared_data, station_indexs)
            
    def __getitem__(self, indexs):
        # import time
        # st = time.time()

        data, station_ids = self.batch_sample(indexs)

        input = data[:,:self.seq_len,:]
        gt = data[:,-(self.pred_len+self.label_len):,:]
        seq_x_mark, seq_x  = input[:,:,:4],input[:,:,4:]
        seq_y_mark, seq_y  = gt[:,:,:4],gt[:,:,4:]
       
        # print(time.time()-st, seq_x.shape)
        # print(seq_x)
        # import pdb
        # pdb.set_trace()
        if self.flag=='test':
            percentile = self.percentiles[station_ids,:,:][:,None,:,:]
            return seq_x, seq_y, seq_x_mark, seq_y_mark, percentile
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def batch_sample(self, indexs): 
        indexs = np.array(indexs, dtype=np.int64)
        indexs = indexs[None] if indexs.ndim==0 else indexs   

        station_ids, s_begin = indexs // self.num_timestamp_input, indexs % self.num_timestamp_input

        if not self.registed_completed:
            if len(self.registed_StationIndex) == len(self.station_names):
                self.registed_completed = True
                print('dataset has been fully loaded into shared memory!')

            need_load = [id for id in station_ids if id not in self.registed_StationIndex]
            need_load = list(set(need_load))
            self.multi_process_load(need_load)

        # sample_indexs = random.choices(self.registed_StationIndex, k=1024)
        s_end = s_begin + self.seq_len + self.pred_len

        seq_x = np.array([copy.deepcopy(self.shared_data_np[sid, st:et,:]) for (sid, st, et) in zip(station_ids, s_begin, s_end)])

        return seq_x, station_ids

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1

        return self.num_timestamp_input * self.num_station
 
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
