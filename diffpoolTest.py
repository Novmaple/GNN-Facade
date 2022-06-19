from dataProcess.dataLoad import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, dense_diff_pool, Linear

n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n : 2 * n]
train_dataset = dataset[2 * n :]


class GNN(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, normalize=False, lin=True
    ):
        super(GNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        for step in range(len(self.convs)):
            x = self.bns[step](F.relu(self.convs[step](x, adj, mask)))
        return x


class DiffPool(torch.nn.Module):
    def __init__(self):
        super(DiffPool, self).__init__()
        self.gnn1_pool = GNN(1, 64, 64)
        self.gnn1_embed = GNN(1, 64, 64)
        self.gnn2_pool = GNN(64, 64, 64)
        self.gnn2_embed = GNN(64, 64, 64, lin=False)
        self.gnn3_embed = GNN(64, 64, 64, lin=False)
        self.lin1 = Linear(64, 64)
        self.lin2 = Linear(64, 2)

    def forward(self, data, mask=None):
        x, adj = data.x, data.pos
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        # x_1 = s_0.t() @ z_0
        # adj_1 = s_0.t() @ adj_0 @ s_0
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        x = self.gnn3_embed(x, adj)
        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffPool().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train(epoch):
    model.train()
    loss_all = 0
    for data in train_dataset:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=1)
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


best_val_acc = test_acc = 0
for epoch in range(1, 151):
    train_loss = train(epoch)
    val_acc = test(val_dataset)
    if val_acc > best_val_acc:
        test_acc = test(test_dataset)
        best_val_acc = val_acc
    print(
        f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, "
        f"Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}"
    )
