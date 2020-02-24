from net import Net, F
from data_processing import get_data
import torch
import torch.optim as optim


train_set, test_set = get_data()

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
n_epochs = 25

net.train()

for epoch in range(n_epochs):
    full_loss = 0
    n_loss = 0
    for data in train_set:
        input, target = data
        optimizer.zero_grad()
        output = net(input)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        full_loss += loss.item()
        n_loss += 1

    print(f'{epoch}: {full_loss / n_loss}')

torch.save(net.state_dict(), '../model/model.pth')

# Testing the model
net = Net()
net.load_state_dict(torch.load('../model/model.pth'))
net.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in test_set:
        input, target = data
        output = net(input)
        for idx, i in enumerate(output):
            if torch.argmax(i) == target[idx]:
                correct += 1
            total += 1

print(f'Accuracy: {round(correct / total, 3)}')
