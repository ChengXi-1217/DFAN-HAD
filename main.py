import os
import time
import test
import train
import torch
import model
import argparse
import numpy as np
import matplotlib
import scipy.io as sio
import evaluate as Eva
import matplotlib.pyplot as plt
import torch.utils.data as Data
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
import evaluate as Eva
from multiprocessing import Process



def HAD(args):

    start_time = time.time()
    data = args.data  # 2D
    gt = args.GT
    latent_layer_dim = args.latent_layer_dim

    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DFAN = model.OrthoAE_Unet(args.input_dim, latent_layer_dim, args)
    DFAN.cuda()

    enc_fea, Pretrain_DFAN, args = train.pre_DFAN(data, DFAN, args)
    train_loss, Trian_DFAN, weight = train.DFAN(data, Pretrain_DFAN, args)
    lat_fea, output = test.DFAN(data, Trian_DFAN, args)

    rec_result = np.array(output.cpu(), dtype=float)
    rec_result = min_max_normal.fit_transform(rec_result)
    AD_result, _ = model.RX(rec_result-data)

    end_time = time.time()
    runing_time = end_time - start_time
    PD_PF_auc, PF_tau_auc, PF, _, _ = \
        Eva.false_alarm_rate(gt.reshape((-1, 1)), AD_result.reshape((-1, 1)))

    print('Dataset:', args.dataset_name)
    print('AUC: PD_PF_auc=%.5f / PF_tau_auc=%.5f' % (PD_PF_auc, PF_tau_auc))
    print('runing-time=%.5f' % runing_time)

    AD_result = AD_result.reshape((args.length, args.width))

    if os.path.exists(args.save_dir):
        pass
    else:
        os.makedirs(args.save_dir)

    S_save = dict()
    S_save['gt'] = gt
    S_save['det'] = AD_result
    result_file_name = args.dataset_name + str('.mat')
    sio.savemat(os.path.join(args.save_dir, result_file_name), S_save)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./Data/abu-airport-4.mat")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pre_epochs', type=int, default=200)
    parser.add_argument('--itetations', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=40000)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--latent_layer_dim', type=int, default=64)
    parser.add_argument('--anomal_prop', type=float, default=0.003)
    parser.add_argument('--bandwidth', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=0.3)  # DBSCAN参数， 邻域半径：  其值越大，聚类的类别数量越少
    parser.add_argument('--min_samples', type=int, default=3)  # DBSCAN参数，  邻域半径最少点：其值越大，聚类的类别数量越少，

    args = parser.parse_args()

    dataset = sio.loadmat(args.data_dir)
    path, file = os.path.split(args.data_dir)
    file_name = file[:-4]
    args.dataset_name = file_name
    HSI_3D = np.array(dataset['data'], dtype=float)
    GT = np.array(dataset['map'], dtype=float)
    length, width, bands = HSI_3D.shape
    HSI_2D = np.reshape(HSI_3D, (length * width, bands))
    min_max_normal = preprocessing.MinMaxScaler()
    args.data = min_max_normal.fit_transform(HSI_2D)

    args.input_dim = bands
    args.length = length
    args.width = width
    args.GT = GT

    dir = os.getcwd()
    save_dir = os.path.join(dir, 'Result')
    args.save_dir = save_dir

    HAD(args)



