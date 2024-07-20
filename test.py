import torch
from torch import nn, optim
import torch.utils.data as Data


def DFAN(data, model, args):
    print("Third Stage: Anomaly Detection")
    model.eval()
    data = torch.from_numpy(data).float()
    tset_load_data = Data.DataLoader(data, batch_size=40000, shuffle=False)
    with torch.no_grad():
        for batch_idx, data in enumerate(tset_load_data):
            lat_fea, output = model(data.cuda(), args)
    return lat_fea, output