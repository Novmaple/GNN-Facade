import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx

# 通过邻接矩阵生成图
# 有向图
nx.from_numpy_matrix(np.array(data), create_using=nx.DiGraph)
# 无向图
nx.from_numpy_matrix(np.array(data))
# 修改图的属性
G.graph["day"] = "Monday"
# 修改节点属性
G.add_nodes_from([3], time="2pm")
G.nodes[1]["room"] = 714
G.nodes.data()
# 修改边的属性
G.add_edges_from([(3, 4), (4, 5)], color="red")
G.add_edges_from([(1, 2, {"color": "blue"}), (2, 3, {"weight": 8})])
G[1][2]["weight"] = 4.7
G.edges[3, 4]["weight"] = 4.2
# 绘制
def draw(Data):  # type(Data) = <class 'torch_geometric.data.data.Data'>
    G = to_networkx(Data)
    nx.draw(G)
    plt.show()
