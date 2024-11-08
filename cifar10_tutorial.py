import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets
import csv
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from neuralnets import Backprop, ConvNet
from torch.nn.functional import softmax

def filter_data(dataset, labels_to_keep):
    filtered_data = []
    filtered_targets = []

    for data, target in zip(dataset.data, dataset.targets):
        if target in labels_to_keep:
            filtered_data.append(data)
            filtered_targets.append(target)

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
    batch_labels = torch.tensor([label_mapping[label.item()] for label in batch_labels], dtype=torch.long)
    return batch_labels
def filter_split_and_create_dataloaders(all_images, all_labels, x, a, y, train_batch_size=100, test_batch_size=200):
    # Step 1: Filter images and labels based on the mask condition
    mask = (all_labels == (x * 2) + 1 + (100 * a)) | (all_labels == (x * 2) + 2 + (100 * a) + y * 2)
    filtered_images, filtered_labels = all_images[mask], all_labels[mask]

    train_images, train_labels, test_images, test_labels = [], [], [], []

    # Step 2: Split filtered images and labels into training and testing datasets
    for label in np.unique(filtered_labels):
        indices = np.where(filtered_labels == label)[0]
        np.random.shuffle(indices)

        # Select 600 training images and 100 testing images per label
        train_indices = indices[:600]
        test_indices = indices[600:700]

        train_images.append(filtered_images[train_indices])
        train_labels.append(filtered_labels[train_indices])
        test_images.append(filtered_images[test_indices])
        test_labels.append(filtered_labels[test_indices])

    # Step 3: Concatenate lists into numpy arrays
    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)
    test_images = np.concatenate(test_images)
    test_labels = np.concatenate(test_labels)

    # Step 4: Convert to tensors and create DataLoaders
    train_dataloader = DataLoader(TensorDataset(torch.tensor(train_images), torch.tensor(train_labels)),
                                  batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(torch.tensor(test_images), torch.tensor(test_labels)),
                                 batch_size=test_batch_size, shuffle=False)

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    directory = os.path.join('ImgNet', 'Imagenet32_train')

    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = ConvNet()

    learner = Backprop(
        net=net,
        step_size=0.01,
        opt="sgd",
        loss='nll',
        weight_decay=0.0,
        device=device,
        momentum=0.9
    )

    tasknumber = 0

    for y in range(5):  # for more than 2000 pair runs5
        a = 0
        for filename in sorted(os.listdir(directory)):
            data = np.load(os.path.join(directory, filename))
            all_images = data['images']
            all_labels = data['labels']
            all_images = preprocess_images(all_images)

            for x in range(44):
                net.layers[-1].weight.data *= 0
                net.layers[-1].bias.data *= 0
                train_dataloader, test_dataloader = filter_split_and_create_dataloaders(all_images, all_labels, x, a, y,100, 200)
                for epoch in range(250):  # 250
                    for batch_images, batch_labels in train_dataloader:
                        # Move images and labels to the GPU if available
                        batch_images, batch_labels = batch_images.to(device), process_batch_labels(batch_labels).to(device)
                        loss, network_output = learner.learn(x=batch_images, target=batch_labels)

                # Save accuracy after each task
                with open('accuracy_results.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    tasknumber += 1
                    # Evaluation on test set
                    with torch.no_grad():
                        batch_images, batch_labels = next(iter(test_dataloader))
                        # Move images and labels to GPU for inference
                        batch_images, batch_labels = batch_images.to(device), process_batch_labels(batch_labels).to(device)
                        outputs, _ = net.predict(batch_images)
                        predicted_labels = torch.argmax(outputs, dim=1)
                        correct = (predicted_labels == batch_labels).sum().item()
                        total = batch_labels.size(0)
                    # Calculate and log accuracy
                    accuracy = 100 * correct / total
                    print(f'Accuracy of the network: {accuracy:.2f} % in Tasknumber : {tasknumber}')
                    writer.writerow([tasknumber, accuracy])
            a += 1

