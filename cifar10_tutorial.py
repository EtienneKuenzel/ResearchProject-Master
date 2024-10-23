import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
import csv
# Define a Convolutional Neural Network
class Nettest(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional + Max-Pooling Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=32 * 21 * 21, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=2)

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

def filter_data(dataset, labels_to_keep):
    filtered_data = []
    filtered_targets = []

    for data, target in zip(dataset.data, dataset.targets):
        if target in labels_to_keep:
            filtered_data.append(data)
            filtered_targets.append(target)

    # Convert the list of numpy arrays to a single numpy array
    filtered_data = np.array(filtered_data)
    filtered_targets = np.array(filtered_targets)

    # Create a new CIFAR10 dataset with the filtered data and targets
    new_dataset = datasets.CIFAR100(root=dataset.root, train=dataset.train,
                                   transform=dataset.transform, target_transform=dataset.target_transform,
                                   download=False)
    new_dataset.data = filtered_data
    new_dataset.targets = filtered_targets.tolist()

    return new_dataset


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    # Define the batch size
    batch_size = 10
    # Define transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Load the CIFAR-100 dataset
    trainsets = []
    testsets = []
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    for x in range(1000): #max 50
        labels_to_keep = [(0+2*x)%100, (1+2*x)%100]
        # Apply the filter to the train and test sets and add their dataloader to the array
        trainsets.append(torch.utils.data.DataLoader(filter_data(trainset, labels_to_keep), batch_size=batch_size, shuffle=True, num_workers=2))
        testsets.append(torch.utils.data.DataLoader(filter_data(testset, labels_to_keep), batch_size=batch_size, shuffle=True, num_workers=2))
        # get some random training images
        #imshow(torchvision.utils.make_grid(next(iter(trainsets[-1]))[0]))
    # Instantiate the network
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net()
    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    a= 0
    for trainloader, testloader in zip(trainsets, testsets):
        a+=1
        for epoch in range(5):  # 20 good time to get used to data pair
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                labels = (labels == labels.max()).long()  # Convert labels to binary as required

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        print('Finished Training' + str())

        with open('accuracy_results.csv', mode='a', newline='') as file:
            writer = csv.writer(file)

            # Check accuracy on the whole dataset
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    labels = (labels == labels.max()).long()
                    _, predicted = torch.max(net(images), 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Accuracy of the network in differentiating the two images: {accuracy:.2f} %')

            # Write the accuracy to the CSV file
            writer.writerow([a, accuracy])
