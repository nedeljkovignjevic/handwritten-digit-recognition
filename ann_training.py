from net import Net, F
from data_processing import get_data
import torch
import torch.optim as optim


train_set, test_set = get_data()

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
n_epochs = 15

net.train()

for epoch in range(n_epochs):
    full_loss = 0
    n_loss = 0
    for data in train_set:
        input, target = data

        net.zero_grad()
        output = net(input.view(-1, 784))

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        full_loss += loss.item()
        n_loss += 1

    print(f'{epoch}: {full_loss / n_loss}')

torch.save(net, 'model/model.pth')

