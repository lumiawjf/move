import socket
import jsons
import ipfshttpclient
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms
from io import BytesIO
from typing import cast
import numpy as np
import logging
from threading import Thread
from copy import deepcopy
import pprint

def ipfs_cat(hash_ipfs):
    api = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
    line = str(api.cat(hash_ipfs))
    line = line.replace('\'', '\"')
    line = line[2:len(line)-1]
    # print("The length of line is %d" %len(line))
    weight = []
    try:
        weight = jsons.loads(line)
    except jsons.exceptions.DecodeError:
        with open("jsons.txt","a") as f:
            f.write(line)
    return weight
def parameters_to_weights(parameters):
    """Convert parameters object to NumPy weights."""
    return [bytes_to_ndarray(tensor) for tensor in parameters]

def bytes_to_ndarray(tensor):
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(bytes(tensor))
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(np.ndarray, ndarray_deserialized)

def weights_to_parameters(weights):
    """Convert NumPy weights to parameters object."""
    tensors = [ndarray_to_bytes(ndarray) for ndarray in weights]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

path = './hash.txt'
with open(path) as f:
    IPFS = f.readlines()
weight1 = ipfs_cat(IPFS[0].replace("\n", ""))
weight2 = ipfs_cat(IPFS[1].replace("\n", ""))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ae = Net().to(device)
init_weight = model_ae.state_dict()
# assert 命令 true则无事发生 false则报错
assert weight1['tensors'] != weight2['tensors'], "two weights are the same."
# 聚合算法
for i, layer in enumerate(init_weight.keys()):
    init_weight[layer] = torch.from_numpy(parameters_to_weights(weight1['tensors'])[i]) / 2 + torch.from_numpy(
        parameters_to_weights(weight2['tensors'])[i]) / 2
path = './weight2'
with open(path, 'w+') as f:
    f.write(str(jsons.dumps(init_weight)))
