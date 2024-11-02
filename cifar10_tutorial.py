import torch
import pandas as pd
import os
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
import csv
import pickle
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from neuralnets import StandardNetCC100, StandardNetCIN

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
    new_dataset = datasets.CIFAR100(root=dataset.root, train=dataset.train,transform=dataset.transform, target_transform=dataset.target_transform,download=False)
    new_dataset.data = filtered_data
    new_dataset.targets = filtered_targets.tolist()

    return new_dataset

def preprocess_images(images):
    images = images.reshape(-1, 3, 32, 32)  # Reshape to (N, C, H, W)
    images /= 255.0  # Normalize if required
    return images
def create_dataloader(images, labels, batch_size=64):
    tensor_images = torch.tensor(images)
    tensor_labels = torch.tensor(labels)
    dataset = TensorDataset(tensor_images, tensor_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
def process_batch_labels(batch_labels):
    unique_labels = torch.unique(batch_labels)
    label_mapping = {unique_labels[0].item(): 0, unique_labels[1].item(): 1}
    # Use vectorized operations to map the labels
    batch_labels = torch.tensor([label_mapping[label.item()] for label in batch_labels], dtype=torch.long)
    return batch_labels




if __name__ == '__main__':
    directory = 'ImgNet/Imagenet32_train/'  # Replace with your file path
    # Instantiate the network
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = StandardNetCIN().to(device)  # Make sure to move the model to the appropriate device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    tasknumber = 0
    for y in range(5):#5 for more than 2000 pair runs
        a=0
        print("Round" + str(1))
        for filename in os.listdir(directory):
            data = np.load(os.path.join(directory, filename))
            all_images = data['images']  # Shape: (num_images, height, width, channels)
            all_labels = data['labels']  # Shape: (num_images,)

            # Preprocess images
            all_images = preprocess_images(all_images)

            for x in range(44):#44 for more than 2000 pair runs
                mask = (all_labels == (x * 2) + 1 + (100 * a)) | (all_labels == (x * 2) + 2 + (100 * a)+y*2)
                filtered_images, filtered_labels = all_images[mask], all_labels[mask]
                train_images, train_labels, test_images, test_labels = [], [], [], []

                # Split images into training and testing datasets
                for label in np.unique(filtered_labels):
                    indices = np.where(filtered_labels == label)[0]
                    np.random.shuffle(indices)

                    # Select training images (600 for each label)
                    train_indices = indices[:600]
                    train_images.append(filtered_images[train_indices])
                    train_labels.append(filtered_labels[train_indices])

                    # Select testing images (100 for each label)
                    test_indices = indices[600:700]
                    test_images.append(filtered_images[test_indices])
                    test_labels.append(filtered_labels[test_indices])

                # Convert lists to numpy arrays
                train_images = np.concatenate(train_images)
                train_labels = np.concatenate(train_labels)
                test_images = np.concatenate(test_images)
                test_labels = np.concatenate(test_labels)

                # Create DataLoaders for training and testing
                train_dataloader = create_dataloader(train_images, train_labels, batch_size=100)
                test_dataloader = create_dataloader(test_images, test_labels, batch_size=200)
                for epoch in range(1):  #250
                    for batch_images, batch_labels in train_dataloader:
                        batch_labels = process_batch_labels(batch_labels)

                        optimizer.zero_grad()
                        outputs = net(batch_images)
                        loss = criterion(outputs, batch_labels)
                        loss.backward()
                        optimizer.step()
                with open('accuracy_results.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    tasknumber +=1
                    correct, total = 0, 0
                    with torch.no_grad():
                        for batch_images, batch_labels in test_dataloader:
                            #imshow(torchvision.utils.make_grid(batch_images))
                            batch_labels = process_batch_labels(batch_labels)
                            _, predicted = torch.max(net(batch_images), 1)
                            correct += (predicted == batch_labels).sum().item()
                            total += batch_labels.size(0)
                    accuracy = 100 * correct / total
                    print(f'Accuracy of the network: {accuracy:.2f} % in Tasknumber : {tasknumber}')
                    writer.writerow([tasknumber, accuracy])
            a+=1

    #Continual CIFAR
    """# Define the batch size
    batch_size = 10
    # Define transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Load the CIFAR-100 dataset
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)


    # Instantiate the network
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = StandardNetCC100()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    a= 0
    labels_to_keep = [0, 1, 2, 3, 4]
    for x in range(19):
        a+=1
        # Apply the filter to the train and test sets and add their dataloader to the array
        trainloader = torch.utils.data.DataLoader(filter_data(trainset, labels_to_keep), batch_size=batch_size, shuffle=True, num_workers=2)
        testloader  = torch.utils.data.DataLoader(filter_data(testset, labels_to_keep), batch_size=batch_size, shuffle=True, num_workers=2)
        labels_to_keep.extend(range(labels_to_keep[-1] + 1, labels_to_keep[-1] + 6))
        #imshow(torchvision.utils.make_grid(next(iter(trainsets[-1]))[0]))
        for epoch in range(1):  #
            for inputs, labels  in trainloader:#training of network
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        with open('accuracy_results.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in testloader:
                    _, predicted = torch.max(net(images), 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            accuracy = 100 * correct / total
            print(f'Accuracy of the network: {accuracy:.2f} %')
            writer.writerow([a, accuracy])"""
