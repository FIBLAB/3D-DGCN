import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):

    def __init__(self,
                 division='regular',
                 C=2,
                 K=9,
                 T=5):
        super(Model, self).__init__()

        self.T_SLOT = 24
        self.K = K
        self.T = T
        self.Gamma = 1

        A = np.load('path/' + division + '_path.npy')
        A_0 = A[:, :, :, 0]
        A_sum = np.sum(A[:, :, :, 1:], axis=3)
        for i in range(1, K + 1):
            A[:, :, :, i] = A_sum
        self.A = torch.Tensor(A)

        feature = np.load('poi/' + division + '_feature.npy').astype(np.float32)

        feature_size = np.size(feature, 1)
        V = np.size(feature, 0)
        self.F = torch.Tensor(feature)
        self.A_mean = torch.Tensor(np.mean(A_0 + A_sum, axis=0)).permute(1, 0).contiguous()
        self.I = torch.ones(V, V, 1)

        self.networks = nn.ModuleList([
            DGCN(C_in=C, C_out=32, kernel_size=(2 * self.Gamma + 1, K + 1)),
            DGCN(C_in=32, C_out=64, kernel_size=(2 * self.Gamma + 1, K + 1)),
            DGCN(C_in=64, C_out=32, kernel_size=(2 * self.Gamma + 1, K + 1)),
            DGCN(C_in=32, C_out=16, kernel_size=(2 * self.Gamma + 1, K + 1)),
        ])
        
        self.semi1 = nn.Linear(feature_size, 16)
        self.semi2 = nn.Linear(16, K)
        self.LeakyReLU = nn.LeakyReLU()
        self.Softmax = nn.Softmax()

        self.temporal_conv = nn.Conv2d(16, C, kernel_size=(T, 1))

    def forward(self, x, x_t, idx):

        N, C, T, V = x.size()
        last = x[:, :, -1, :]

        Y1 = self.LeakyReLU(torch.mm(self.A_mean, self.semi1(self.F)))
        Y2 = self.LeakyReLU(torch.mm(self.A_mean, self.semi2(Y1)))
        Y = torch.index_select(Y2, 0, idx)
        Z = self.Softmax(Y2)
        A_K = self.A * torch.cat((self.I, Z.repeat(V, 1, 1)), dim=2).repeat(self.T_SLOT * 2, 1, 1, 1)

        x_t = x_t.view(N * T)
        x_A = torch.index_select(A_K, 0, x_t)
        x_A = x_A.view(N, T, V, V, -1)

        for dgcn in self.networks:
            x = dgcn(x, x_A)

        x = self.temporal_conv(x)
        x = x.view(N, -1, V) + last

        return x, Y


class DGCN(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size):
        super(DGCN, self).__init__()

        self.K = kernel_size[1] - 1
        self.C_out = C_out
        self.conv = nn.Sequential(nn.BatchNorm3d(C_in),
                                  nn.ReLU(inplace=True),
                                  nn.Conv3d(C_in,
                                            C_out,
                                            kernel_size=(kernel_size[0], 1, kernel_size[1]),
                                            padding=(1, 0, 0),
                                            stride=(1, 1, 1),
                                            bias=True),
                                  nn.BatchNorm3d(C_out),
                                  nn.Dropout3d(p=0.2, inplace=True))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x_A):

        N, C, T, V = x.size()

        x_reshape = x.permute(0, 2, 1, 3).contiguous().view(N * T, C, V)
        A_reshape = x_A.view(N * T, V, V * (self.K + 1)).contiguous()
        x = torch.bmm(x_reshape, A_reshape).view(N, T, C, V, (self.K + 1)).permute(0, 2, 1, 3, 4).contiguous()
        x = self.conv(x).view(N, self.C_out, T, V)

        return self.relu(x)
