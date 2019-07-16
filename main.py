from gcn import Model
from dataset import PopulationDataset
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from math import sqrt
from time import time


def train(model, train_loader, optimizer, criterion, epoch, label, label_idx, label_weight, theta):
    model.train()
    train_loss = 0
    for batch_idx, (data, x_t, target) in enumerate(train_loader):
        data, x_t, target = Variable(data).cuda(), Variable(x_t).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output, y = model(data, x_t, label_idx)
        loss1 = criterion(output, target)
        loss2 = nn.functional.cross_entropy(y, label, weight=label_weight)
        loss = loss1 + loss2 * theta
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print ("Epoch: %d [%d/%d (%.0f%%)]\tLoss: %.6f\tLabel Loss: %.6f" % (
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), sqrt(loss1.data[0]), loss2.data[0]))
        train_loss += loss1.data[0] * len(data)
    return sqrt(train_loss / len(train_loader.dataset))


def val(model, val_loader, criterion, epoch, label_idx):
    model.eval()
    val_loss = 0
    for batch_idx, (data, x_t, target) in enumerate(val_loader):
        data, x_t, target = Variable(data).cuda(), Variable(x_t).cuda(), Variable(target).cuda()
        output, _ = model(data, x_t, label_idx)
        loss = criterion(output, target)
        val_loss += loss.data[0] * len(data)
    print ("\nEpoch: %d \tVal Loss: %.6f" % (
        epoch, sqrt(val_loss / len(val_loader.dataset))))
    return sqrt(val_loss / len(val_loader.dataset))


def test(model, test_loader, criterion, epoch, label_idx):
    model.eval()
    test_loss = 0
    for batch_idx, (data, x_t, target) in enumerate(test_loader):
        data, x_t, target = Variable(data).cuda(), Variable(x_t).cuda(), Variable(target).cuda()
        output, _ = model(data, x_t, label_idx)
        loss = criterion(output, target)
        test_loss += loss.data[0] * len(data)
    print ("Epoch: %d \tTest Loss: %.6f\n" % (
        epoch, sqrt(test_loss / len(test_loader.dataset))))
    return sqrt(test_loss / len(test_loader.dataset))


def main():
    parser = argparse.ArgumentParser(description='3D-DGCN')
    parser.add_argument('--batch', type=int, default=64, metavar='B',
                        help='batch size for training')
    parser.add_argument('--test-batch', type=int, default=32, metavar='TB',
                        help='batch size for testing')
    parser.add_argument('--epoch', type=int, default=1600, metavar='E',
                        help='number of iterations')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--division', type=str, default='regular', metavar='D',
                        help='the spatial division (regular or irregular)')
    parser.add_argument('--K', type=int, default='9', metavar='K',
                        help='number of partitions')
    parser.add_argument('--C', type=int, default='2', metavar='C',
                        help='number of input channels')
    parser.add_argument('--T', type=int, default='5', metavar='T',
                        help='number of time intervals in a training sample')
    parser.add_argument('--theta', type=float, default='10', metavar='T',
                        help='hyper-parameter in loss function')
    parser.add_argument('--load', type=int, default='0', metavar='L',
                        help='load checkpoint/epoch_x.tar (x>0)')
    args = parser.parse_args()

    train_loader = torch.utils.data.DataLoader(PopulationDataset(division=args.division,
                                                                 T=args.T,
                                                                 type='train'),
                                               batch_size=args.test_batch,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(PopulationDataset(division=args.division,
                                                               T=args.T,
                                                               type='val'),
                                             batch_size=args.test_batch,
                                             shuffle=False)

    test_loader = torch.utils.data.DataLoader(PopulationDataset(division=args.division,
                                                                T=args.T,
                                                                type='test'),
                                              batch_size=args.test_batch,
                                              shuffle=False)

    model = Model(division=args.division,
                  C=args.C,
                  K=args.K,
                  T=args.T)

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    model = model.cuda()
    
    model.A = Variable(model.A).cuda()
    model.F = Variable(model.F).cuda()
    model.A_mean = Variable(model.A_mean).cuda()
    model.I = Variable(model.I).cuda()

    label = np.load('poi/' + args.division + '_label.npy').astype(np.int64)
    label_idx = np.load('poi/' + args.division + '_idx.npy').astype(np.int64)
    label_weight = np.load('poi/' + args.division + '_weight.npy').astype(np.float32)
    
    label = torch.from_numpy(label[label_idx])
    label_idx = torch.from_numpy(label_idx)
    label_weight = torch.from_numpy(label_weight)
    label = Variable(label).cuda()
    label_idx = Variable(label_idx).cuda()
    label_weight = Variable(label_weight).cuda()

    if args.load > 0:
        model.load_state_dict(torch.load('checkpoint/epoch_' + str(args.load) + '.tar'))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(size_average=True)
    loss = np.zeros((3, args.epoch), dtype=np.float32)
    val_loss_min = 100
    test_loss_min = 100

    for epoch in range(1 + args.load, args.epoch + 1 + args.load):
        start = time()
        train_loss = train(model, train_loader, optimizer, criterion, epoch, label, label_idx, label_weight, args.theta)
        val_loss = val(model, val_loader, criterion, epoch, label_idx)
        test_loss = test(model, test_loader, criterion, epoch, label_idx)
        stop = time()
        print ("Time used: %.0f" % (stop - start))
        loss[0, epoch - args.load - 1] = train_loss
        loss[1, epoch - args.load - 1] = val_loss
        loss[2, epoch - args.load - 1] = test_loss
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            test_loss_min = test_loss
        if epoch % 100 == 0:
            torch.save(model.state_dict(), 'checkpoint/epoch_' + str(epoch) + '.tar')

    np.save('checkpoint/loss.npy', loss)
    print("Val RMSE: %.4f \tTest RMSE: %.4f\n" % (
        val_loss_min, test_loss_min))


if __name__ == '__main__':
    main()
