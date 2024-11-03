import torch
import torch.nn as nn
import torch.nn.functional as F
class StandardNetCC100(nn.Module):
    def __init__(self):
        super(StandardNetCC100, self).__init__()
        # Convolutional + Max-Pooling Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)  # kernel size reduced to 3
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # pool stride changed to 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # pool stride changed to 2
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # pool stride changed to 2

        # Fully Connected Layers
        # Adjusted in_features based on pooling and reduced spatial size
        self.fc1 = nn.Linear(in_features=32 * 2 * 2, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=100)

        # Optional Dropout Layer to improve generalization
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





class StandardNetCIN(nn.Module):
    def __init__(self):
        super(StandardNetCIN, self).__init__()

        # Convolutional + Max-Pooling Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=128 * 21 * 21, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
