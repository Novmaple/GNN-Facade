from dataProcess.dataLoad import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, Linear

n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n : 2 * n]
train_dataset = dataset[2 * n :]


class GCN(torch.nn.Module):
    def __init__(self, input, hidden, output):
        super(GCN, self).__init__()
        self.input = input
        self.hidden = hidden
        self.output = output
        self.conv1 = GCNConv(input, hidden)
        self.conv2 = GCNConv(hidden, output)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # self.embedding=Node2Vec(embedding_dim=128, walk_length=20,
        #                         context_size=10, walks_per_node=10,
        #          num_negative_samples=1, p=1, q=1, sparse=True)
        self.filter1 = GCNConv(1, 16)
        self.filter2 = GCNConv(16, 16)
        self.pool = global_mean_pool
        self.linear = Linear(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # embed=self.embedding(edge_index)
        out = F.relu(self.filter1(x, edge_index))
        out = F.relu(self.filter2(out, edge_index))
        out = self.pool(out, torch.tensor([0]))
        out = self.linear(out)
        return F.log_softmax(out, dim=1)


device = torch.device("cpu")
model = Classifier().to(device)
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
    return correct / len(loader)


best_val_acc = test_acc = 0
for epoch in range(30):
    train_loss = train(epoch)
    val_acc = test(val_dataset)
    if val_acc > best_val_acc:
        test_acc = test(test_dataset)
        best_val_acc = val_acc
    print(
        f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, "
        f"Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}"
    )
