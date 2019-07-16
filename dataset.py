import torch
from torch.utils.data import Dataset
import json
import numpy as np


class PopulationDataset(Dataset):

    def __init__(self,
                 division='regular',
                 T=5,
                 type='train'):
        len_t = 24 * (31 + 31 + 30)
        N = len_t - T
        N_train = int(N * 0.8)
        N_val = int(N * 0.1)
        N_test = int(N * 0.1)
        weekend = [i for i in range(1, 93, 7)] + [i for i in range(2, 93, 7)] + [4, 66]
        T_SLOT = 24
        C = 2
        if division == 'regular':
            H = 16
            W = 8
            num_node = H * W
            raw_data = json.load(open("flow/flow_bike_nyc_regular.json"))
        elif division == 'irregular':
            num_node = 82
            raw_data = json.load(open("flow/flow_bike_nyc_irregular.json"))

        dataset = np.zeros((C, len_t, num_node), dtype=np.float32)
        data1 = raw_data['inflow']
        data2 = raw_data['outflow']

        for node_no in range(num_node):
            dataset[0, :, node_no] = np.array(data1[str(node_no)])
            dataset[1, :, node_no] = np.array(data2[str(node_no)])

        if type == 'train':
            xy = np.zeros((N_train, C, T + 1, num_node), dtype=np.float32)
            x_t = np.zeros((N_train, T), dtype=np.int64)
            for t in range(N_train):
                xy[t, :, :, :] = dataset[:, t:t + T + 1, :]
                if (t + T + 1) // T_SLOT + 1 in weekend:
                    x_t[t, :] = np.arange(t, t + T) % T_SLOT + T_SLOT
                else:
                    x_t[t, :] = np.arange(t, t + T) % T_SLOT

        elif type == 'val':
            xy = np.zeros((N_val, C, T + 1, num_node), dtype=np.float32)
            x_t = np.zeros((N_val, T), dtype=np.int64)
            for t in range(N_train, N_train + N_val):
                xy[t - N_train, :, :, :] = dataset[:, t:t + T + 1, :]
                if (t + T + 1) // T_SLOT + 1 in weekend:
                    x_t[t - N_train, :] = np.arange(t, t + T) % T_SLOT + T_SLOT
                else:
                    x_t[t - N_train, :] = np.arange(t, t + T) % T_SLOT

        elif type == 'test':
            xy = np.zeros((N_test, C, T + 1, num_node), dtype=np.float32)
            x_t = np.zeros((N_test, T), dtype=np.int64)
            for t in range(N_train + N_val, N_train + N_val + N_test):
                xy[t - N_train - N_val, :, :, :] = dataset[:, t:t + T + 1, :]
                if (t + T + 1) // T_SLOT + 1 in weekend:
                    x_t[t - N_train - N_val, :] = np.arange(t, t + T) % T_SLOT + T_SLOT
                else:
                    x_t[t - N_train - N_val, :] = np.arange(t, t + T) % T_SLOT

        self.x_data = torch.from_numpy(xy[:, :, 0:-1, :])
        self.x_t = torch.from_numpy(x_t)
        self.y_data = torch.from_numpy(xy[:, :, -1, :])
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.x_t[index], self.y_data[index]

    def __len__(self):
        return self.len
