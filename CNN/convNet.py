import torch

class ConvNet(torch.nn.Module):
    def __init__(self, image_size_general_local=32, hidden=64, hidden2=32, num_classes=2):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)

        layer_neurons = 324
        if image_size_general_local == 32:
            layer_neurons = 324
        elif image_size_general_local == 64:
            layer_neurons = 1600

        self.fc1 = torch.nn.Linear(layer_neurons, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x * x  # square activation
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = x * x  # square activation
        x = self.fc2(x)
        x = x * x  # square activation
        x = self.fc3(x)
        return x