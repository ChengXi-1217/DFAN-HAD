import os
import torch
import numpy as np
from torch import nn
from numpy import where
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

def gram_schmidt(input):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u
    output = []
    for x in input:
        for y in output:
            x = x - projection(y, x)
        x = x/x.norm(p=2)
        output.append(x)
    return torch.stack(output)

def initialize_orthogonal_filters(c, h, w):

    if h*w < c:
        n = c//(h*w)
        gram = []
        for i in range(n):
            gram.append(gram_schmidt(torch.rand([h * w, 1, h, w])))
        return torch.cat(gram, dim=0)
    else:
        return gram_schmidt(torch.rand([c, 1, h, w]))


def AAM(data, args):
    threshold = int(args.anomal_prop * np.array(data.shape[0]))
    data = data.cpu().detach().numpy()
    min_max_normal = preprocessing.MinMaxScaler()
    data = min_max_normal.fit_transform(data)
    Clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit(data)
    labels = Clustering.labels_
    n_cluster, cluster_list = np.unique(labels, return_counts=True)
    weight = torch.zeros(len(labels), device=None)
    weight_mask = torch.ones(len(labels))
    all_pix_intra_dis = torch.zeros(len(labels))
    all_pix_inter_dis = torch.zeros(len(labels))
    background_cluster = np.where(cluster_list > threshold)
    background_cluster_index = n_cluster[background_cluster]
    anomaly_cluster = np.where(cluster_list <= threshold)
    anomaly_cluster_index = n_cluster[anomaly_cluster]
    centers = []

    for label in range(len(n_cluster)):
        center = np.mean(data[labels == n_cluster[label]], axis=0)
        centers.append(center)
    centers = torch.tensor(centers)
    background_cluster_center = torch.zeros((len(background_cluster[0]), data.shape[1]))

    for i in range(len(background_cluster_index)):
        cluster_i = torch.tensor(data[labels == background_cluster_index[i]])
        index_i = np.where(labels == background_cluster_index[i])
        cluster_i_center = centers[np.where(n_cluster == background_cluster_index[i])]
        cluster_i_intra_dis = 1 - F.cosine_similarity(cluster_i, cluster_i_center, dim=1)
        cluster_i_wight = cluster_i_intra_dis / torch.sum(cluster_i_intra_dis)
        weight[index_i] = cluster_i_wight
        all_pix_intra_dis[index_i] = cluster_i_intra_dis
        cluster_i_inter = F.cosine_similarity(cluster_i.unsqueeze(1), centers.unsqueeze(0), dim=2)
        cluster_i_inter_sort, _ = torch.sort(cluster_i_inter, dim=1)

        if cluster_i_inter_sort.shape[1] > 1:
            cluster_i_inter_dis = cluster_i_inter_sort[:, -1] - cluster_i_inter_sort[:, -2]
        else:
            cluster_i_inter_dis = torch.zeros((cluster_i_inter_sort.shape[0]))

        all_pix_inter_dis[index_i] = cluster_i_inter_dis
        background_cluster_center[i] = cluster_i_center

    for ii in range(len(anomaly_cluster_index)):
        index_ii = np.where(labels == anomaly_cluster_index[ii])
        weight_mask[index_ii] = 0
        weight[index_ii] = 0

    intra_dis = torch.sum(all_pix_intra_dis) / (data.shape[0] * data.shape[1])
    inter_dis = torch.sum(all_pix_inter_dis) / (data.shape[0] * data.shape[1])
    intra_center_dis_mean = torch.mean(torch.norm(background_cluster_center, dim=1)) / data.shape[1]

    return intra_dis, weight_mask, inter_dis, intra_center_dis_mean


def to_cuda(tensor: torch.Tensor) -> torch.Tensor:
    """
    The method is used to determine whether to copy tensors from the CPU to the GPU for accelerated computation,
    primarily used for device migration.
    """
    return tensor.cuda() if torch.cuda.is_available() else tensor
