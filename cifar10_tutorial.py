import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

# Define a Convolutional Neural Network
class Net(nn.Module):
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
    batch_size = 4

    # Define transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Filter the dataset to only include labels 0 (plane) and 1 (car)

    trainsets = []
    testsets = []
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Labels to keep (0: plane, 1: car)
    for x in range(40): #max 50
        labels_to_keep = [0+2*x, 1+2*x]
        # Apply the filter to the train and test sets and add their dataloader to the array
        trainsets.append(torch.utils.data.DataLoader(filter_data(trainset, labels_to_keep), batch_size=batch_size, shuffle=True, num_workers=2))
        testsets.append(torch.utils.data.DataLoader(filter_data(testset, labels_to_keep), batch_size=batch_size, shuffle=True, num_workers=2))
    # Instantiate the network
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Net()
    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    for trainloader, testloader in zip(trainsets, testsets):
        running_loss = 0.0
        for epoch in range(5):  # loop over the dataset multiple times
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                labels = (labels == labels.max()).long()  # This will be 1 where the condition is true, 0 otherwise

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        print('Finished Training' + str())

        # Save the trained model
        #PATH = './cifar_net.pth'
        #torch.save(net.state_dict(), PATH)
        # Reload the model for inference
        #net = Net()
        #net.load_state_dict(torch.load(PATH))

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

        print(f'Accuracy of the network in differentiating the two images: {100 * correct // total} %')

