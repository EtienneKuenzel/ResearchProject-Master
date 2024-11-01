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
from neuralnets import StandardNet

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

def load_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')  # 'latin1' is often needed for Python 2-serialized data
    images = np.array(batch['data'], dtype=np.float32)  # Adjust key if different
    labels = np.array(batch['labels'], dtype=np.int64)  # Adjust key if different
    return images, labels
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



def save_label_group(batch_number, images, labels):
    """Saves images and labels of a group of 100 labels."""
    file_name = f"label_group_batch_{batch_number}.npz"
    np.savez(file_name, images=images, labels=labels)


def save_filtered_labels(images, labels, target_labels, output_file):
    """Saves images and labels for specified target labels into one file."""
    # Filter images and labels that match the target labels
    filtered_indices = np.isin(labels, target_labels)
    filtered_images = images[filtered_indices]
    filtered_labels = labels[filtered_indices]
    np.savez(output_file, images=filtered_images, labels=filtered_labels)


import os
import numpy as np


def save_filtered_labels(images, labels, target_labels, output_file):
    # Check if file already exists
    if os.path.exists(output_file):
        # Load existing data
        existing_data = np.load(output_file)
        existing_images = existing_data['images']
        existing_labels = existing_data['labels']

        # Append new data to the existing arrays
        images = np.concatenate((existing_images, images), axis=0)
        labels = np.concatenate((existing_labels, labels), axis=0)

    # Filter images and labels based on target labels
    mask = np.isin(labels, list(target_labels))
    filtered_images = images[mask]
    filtered_labels = labels[mask]

    # Save combined data back to the .npz file
    np.savez(output_file, images=filtered_images, labels=filtered_labels)
    print(f"Saved updated {output_file}")


if __name__ == '__main__':
    # Load the dataset
    dataset = 'ImgNet/Imagenet32_train/label_100_to_199_data.npz'  # Replace with your file path
    data = np.load(dataset)

    # Assuming your .npz file has keys 'images' and 'labels'
    all_images = data['images']  # Shape: (num_images, height, width, channels)
    all_labels = data['labels']  # Shape: (num_images,)

    # Preprocess images
    all_images = preprocess_images(all_images)

    for x in range(10):
        mask = (all_labels == 199) | (all_labels == 198)
        combined_dataloader = create_dataloader(all_images[mask], all_labels[mask], batch_size=1)
        print(len(combined_dataloader))
        for batch_images, batch_labels in combined_dataloader:
            imshow(torchvision.utils.make_grid(batch_images))
            print(batch_labels)
            break




    #Continual CIFAR
    # Define the batch size
    batch_size = 10
    # Define transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Load the CIFAR-100 dataset
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)


    # Instantiate the network
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = StandardNet()
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
            writer.writerow([a, accuracy])
