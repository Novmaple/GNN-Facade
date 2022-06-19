from dataProcess.dataLoad import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGPooling, global_max_pool, global_mean_pool
import random
random.shuffle(dataset)
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n : 2 * n]
train_dataset = dataset[2 * n :]
train_loader=DataLoader(train_dataset,batch_size=1)
val_loader=DataLoader(val_dataset,batch_size=1)
test_loader=DataLoader(test_dataset,batch_size=1)

class SAGPoolClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.pool1 = SAGPooling(hidden_dim, 0.5)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.pool2 = SAGPooling(hidden_dim, 0.5)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        self.pool3 = SAGPooling(hidden_dim, 0.5)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        gcn1 = F.relu(self.gcn1(x, edge_index))
        pool1 = self.pool1(gcn1, edge_index)
        global_pool1 = torch.cat(
            [
                global_mean_pool(pool1[0], torch.tensor([0])),
                global_max_pool(pool1[0], torch.tensor([0])),
            ],
            dim=1,
        )
        gcn2 = F.relu(self.gcn2(pool1[0], pool1[1]))
        pool2 = self.pool2(gcn2, pool1[1])
        global_pool2 = torch.cat(
            [
                global_mean_pool(pool2[0], torch.tensor([0])),
                global_max_pool(pool2[0], torch.tensor([0])),
            ],
            dim=1,
        )
        gcn3 = F.relu(self.gcn3(pool2[0], pool2[1]))
        pool3 = self.pool3(gcn3, pool2[1])
        global_pool3 = torch.cat(
            [
                global_mean_pool(pool3[0], torch.tensor([0])),
                global_max_pool(pool3[0], torch.tensor([0])),
            ],
            dim=1,
        )
        readout = global_pool1 + global_pool2 + global_pool3
        y = self.mlp(readout)
        return F.log_softmax(y, dim=1)


device = torch.device("cpu")
model = SAGPoolClassifier(1, 16, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train(epoch):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_loader)


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
for epoch in range(1, 11):
    train_loss = train(epoch)
    val_acc = test(val_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        best_val_acc = val_acc
    print(
        f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, "
        f"Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}"
    )
