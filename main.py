import torch
from data_processing import get_data
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Testing the model
    _, test_set = get_data()
    net = torch.load('model/model.pth')

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_set:
            input, target = data
            output = net(input.view(-1, 784))
            for idx, i in enumerate(output):
                if torch.argmax(i) == target[idx]:
                    correct += 1
                total += 1

    print(f'Accuracy: {round(correct / total, 3)}')

    # Show image and ann output
    for el in test_set:
        # Get 1 batch then break
        data, target = el
        break

    print(torch.argmax(net(data[3].view(-1, 784))[0]))
    plt.imshow(data[3].view(28, 28))
    plt.show()

