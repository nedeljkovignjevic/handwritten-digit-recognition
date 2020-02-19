import torch
from torchvision import transforms, datasets


def get_data():
    train_set = datasets.MNIST('dataset', train=True, download=True,
                               transform=transforms.Compose([transforms.ToTensor()]))

    test_set = datasets.MNIST('dataset', train=False, download=True,
                              transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False)

    return train_loader, test_loader
