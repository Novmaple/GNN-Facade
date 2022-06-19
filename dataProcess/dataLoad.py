import pickle
import torch
from torch_geometric.data import Data

with open("dataProcess/data/x", "rb") as fx:
    x = pickle.load(fx)
    fx.close()
with open("dataProcess/data/y", "rb") as fy:
    y = pickle.load(fy)
    fy.close()
with open("dataProcess/data/edge_index", "rb") as fei:
    edge_index = pickle.load(fei)
    fei.close()
with open("dataProcess/data/edge_attr", "rb") as fea:
    edge_attr = pickle.load(fea)
    fea.close()
with open("dataProcess/data/adj", "rb") as fa:
    adj = pickle.load(fa)
    fa.close()
dataset = []
for i in range(len(x)):
    dataset.append(
        Data(
            x=torch.tensor(x[i], dtype=torch.float),
            y=torch.tensor(y[i]),
            edge_attr=torch.tensor(edge_attr[i]),
            edge_index=torch.tensor(edge_index[i], dtype=torch.long),
            pos=torch.tensor(adj[i]),
        )
    )
