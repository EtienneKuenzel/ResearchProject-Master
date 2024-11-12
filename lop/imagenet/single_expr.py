import torch
import pickle
from tqdm import tqdm
from lop.algos.bp import Backprop
from lop.nets.conv_net import ConvNet
from torch.nn.functional import softmax
from lop.nets.linear import MyLinear
from lop.utils.miscellaneous import nll_accuracy as accuracy
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F


train_images_per_class = 600
test_images_per_class = 100
images_per_class = train_images_per_class + test_images_per_class



# Function to display a batch of images
def show_batch(batch_x, batch_y, num_images_to_show=4, denormalize=False):
    """
    Displays a few images from the given batch with their corresponding labels.

    Parameters:
    - batch_x: Tensor of images (batch_size, channels, height, width)
    - batch_y: Tensor of labels (batch_size)
    - num_images_to_show: Number of images to display from the batch (default: 4)
    - denormalize: Whether to denormalize the image before displaying (default: False)
    """
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]  # ImageNet std

    # Helper function to denormalize
    def denormalize_image(img):
        img = img.clone()
        for i in range(3):  # Assuming RGB images
            img[i] = img[i] * std[i] + mean[i]
        return img

    plt.figure(figsize=(12, 6))

    for i in range(min(num_images_to_show, len(batch_x))):
        img = batch_x[i]
        label = batch_y[i].item() if batch_y[i].dim() == 0 else batch_y[i].argmax().item()  # Handle one-hot labels

        if denormalize:
            img = denormalize_image(img)

        # Convert to PIL image for visualization
        img = T.ToPILImage()(img.cpu())

        plt.subplot(1, num_images_to_show, i + 1)
        plt.imshow(img)
        plt.title(f"Label: {label}")
        plt.axis('off')

    plt.show()


def get_activation(name):
    def hook(model, input, output):
        if name not in activations:
            activations[name] = output.detach().clone()
        else:
            # Concatenate activations across batches
            activations[name] = torch.cat((activations[name], output.detach().clone()), dim=0)

    return hook
def count_dormant_neurons_per_layer(activations, threshold=1e-5):
    """
    Count the number of dormant neurons per layer based on activations.

    Parameters:
    - activations (dict): Dictionary of layer activations.
    - threshold (float): Threshold below which a neuron is considered dormant.

    Returns:
    - dormant_count (dict): Count of dormant neurons per layer.
    """
    dormant_count = 0
    for layer_name, act in activations.items():
        if "fc" in layer_name:
            # Calculate average activations across the batch dimension
            avg_activation = torch.mean(act, dim=0)
            # Count neurons with average activation below the threshold
            dormant_neurons = torch.sum(avg_activation < 500).item()
            dormant_count += dormant_neurons
    print(dormant_count)
    return dormant_count


def load_imagenet(classes=[]):
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, _class in enumerate(classes):
        data_file = 'data/classes/' + str(_class) + '.npy'
        new_x = np.load(data_file)
        x_train.append(new_x[:train_images_per_class])
        x_test.append(new_x[train_images_per_class:])
        y_train.append(np.array([idx] * train_images_per_class))
        y_test.append(np.array([idx] * test_images_per_class))
    x_train = torch.tensor(np.concatenate(x_train))
    y_train = torch.from_numpy(np.concatenate(y_train))
    x_test = torch.tensor(np.concatenate(x_test))
    y_test = torch.from_numpy(np.concatenate(y_test))
    return x_train, y_train, x_test, y_test


def save_data(data, data_file):
    with open(data_file, 'wb+') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    num_tasks = 2
    use_gpu = 1
    mini_batch_size = 100
    run_idx = 3
    data_file = "output.pkl"
    num_epochs = 2

    # Device setup
    dev = torch.device("cuda:0") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
    if use_gpu and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Constants
    examples_per_epoch = train_images_per_class * 2

    # Initialize network
    net = ConvNet()
    #net = MyLinear(input_size=3072, num_outputs=classes_per_task)

    # Initialize learner
    learner = Backprop(
        net=net,
        step_size=0.1,
        opt="sgd",
        loss='nll',
        weight_decay=0,
        to_perturb=(0 != 0),
        perturb_scale=0,
        device=dev,
        momentum=0.9,
    )

    # Load class order
    with open('class_order', 'rb') as f:class_order = pickle.load(f)[run_idx]

    class_order = np.concatenate([class_order] * ((2 * num_tasks) // 1000 + 1))

    # Initialize accuracy tracking
    train_accuracies = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    test_accuracies = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    dormant_neurons = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    # Training loop
    for task_idx in range(num_tasks):
        print("Task : " + str(task_idx))
        x_train, y_train, x_test, y_test = load_imagenet(class_order[task_idx * 2:(task_idx + 1) * 2])
        x_train, x_test = x_train.float(), x_test.float()
        #x_train, x_test = x_train.flatten(1), x_test.flatten(1) for linear network

        x_train, y_train = x_train.to(dev), y_train.to(dev)
        x_test, y_test = x_test.to(dev), y_test.to(dev)

        net.layers[-1].weight.data.zero_()
        net.layers[-1].bias.data.zero_()

        # Epoch loop
        for epoch_idx in range(num_epochs): #tqdm
            example_order = np.random.permutation(examples_per_epoch)
            x_train, y_train = x_train[example_order], y_train[example_order]
            new_train_accuracies = torch.zeros(examples_per_epoch // mini_batch_size, dtype=torch.float)

            for i, start_idx in enumerate(range(0, examples_per_epoch, mini_batch_size)):
                batch_x = x_train[start_idx:start_idx + mini_batch_size]
                batch_y = y_train[start_idx:start_idx + mini_batch_size]
                # show_batch(batch_x, batch_y, num_images_to_show=10, denormalize=True)
                loss, network_output = learner.learn(x=batch_x, target=batch_y)
                new_train_accuracies[i] = accuracy(softmax(network_output, dim=1), batch_y).cpu()
            train_accuracies[task_idx][epoch_idx] = new_train_accuracies.mean()
            new_test_accuracies = torch.zeros(x_test.shape[0] // mini_batch_size, dtype=torch.float)

        # Test loop with neuron activation recording
        activations = {}
        # Register hooks to capture activations for all layers
        hooks = []
        for name, layer in net.named_modules():
            if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
                hooks.append(layer.register_forward_hook(get_activation(name)))

        # Running test batches
        for i, start_idx in enumerate(range(0, x_test.shape[0], mini_batch_size)):
            test_batch_x = x_test[start_idx:start_idx + mini_batch_size]
            test_batch_y = y_test[start_idx:start_idx + mini_batch_size]
            network_output, _ = net.predict(x=test_batch_x)
            new_test_accuracies[i] = accuracy(F.softmax(network_output, dim=1), test_batch_y)

        test_accuracies[task_idx][epoch_idx] = new_test_accuracies.mean()
        # Count dormant neurons
        dormant_neurons[task_idx][epoch_idx] = count_dormant_neurons_per_layer(activations)
        # Remove hooks after counting
        for hook in hooks:
            hook.remove()


    # Final save
    save_data({
        'dormant neurons': dormant_neurons.cpu(),
        'train_accuracies': train_accuracies.cpu(),
        'test_accuracies': test_accuracies.cpu()
    }, data_file)





