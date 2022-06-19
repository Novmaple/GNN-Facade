import pickle
import networkx as nx
import torch
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data

with open("data/x", "rb") as fx:
    x = pickle.load(fx)
    fx.close()
with open("data/y", "rb") as fy:
    y = pickle.load(fy)
    fy.close()
with open("data/edge_index", "rb") as fei:
    edge_index = pickle.load(fei)
    fei.close()
with open("data/edge_attr", "rb") as fea:
    edge_attr = pickle.load(fea)
    fea.close()
graphData = []
adj = []
for i in range(len(x)):
    graphData.append(
        Data(
            x=torch.tensor(x[i], dtype=torch.float),
            y=torch.tensor(y[i]),
            edge_attr=torch.tensor(edge_attr[i]),
            edge_index=torch.tensor(edge_index[i], dtype=torch.long),
        )
    )
for i in range(len(graphData)):
    adj.append(nx.adjacency_matrix(to_networkx(graphData[i])).todense())
print(adj)
with open("data/adj", "wb") as fa:
    pickle.dump(adj, fa)
