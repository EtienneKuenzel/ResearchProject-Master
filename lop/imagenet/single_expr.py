import torch
import pickle
from tqdm import tqdm
from lop.algos.bp import Backprop
from lop.nets.conv_net import ConvNet_vanilla,ConvNet_PAU, ConvNet_TENT
from torch.nn.functional import softmax
from lop.nets.linear import MyLinear
from lop.utils.miscellaneous import nll_accuracy as accuracy
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import time as time

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
        # Store activations
        if name not in activations:
            activations[name] = output.detach().clone()
        else:
            activations[name] = torch.cat((activations[name], output.detach().clone()), dim=0)

        # Store inputs to activations
        if name not in inputs_to_activations:
            inputs_to_activations[name] = input[0].detach().clone()  # input is a tuple
        else:
            inputs_to_activations[name] = torch.cat((inputs_to_activations[name], input[0].detach().clone()), dim=0)
    return hook

def average_activation_input(activations, threshold, model):#Important gives out the average input into the activation functions
    dormant_count = 0
    for layer_name, act in activations.items():
        if "fc" in layer_name:
            dormant_count += torch.sum(torch.mean(act, dim=0) < threshold).item()
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
    num_tasks = 2000
    use_gpu = 1
    mini_batch_size = 100
    run_idx = 3
    data_file = "outputRELU.pkl"
    num_epochs =  250

    # Device setup
    dev = torch.device("cuda:0") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
    if use_gpu and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Constants
    examples_per_epoch = train_images_per_class * 2

    # Initialize network
    net = ConvNet_vanilla()
    #net =ConvNet_PAU()
    #net = ConvNet_TENT()
    #net = MyLinear(input_size=3072, num_outputs=classes_per_task)

    # Initialize learner
    learner = Backprop(
        net=net,
        step_size=0.01,
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
    test_accuracies = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    dormant_neurons = torch.zeros(num_tasks, 4000)
    historical_accuracies = torch.zeros(num_tasks, 100)
    training_time = 0
    input_distribution = torch.zeros(num_tasks, 3,3,100)#numtasks, 3=layer, 3=CurrentTask+OOD(Next Task)+Adveserial Attack,100=Datapoints
    weight_layer = torch.zeros((num_tasks, 2, 128))
    bias_layer = torch.zeros(num_tasks, 2)



    # Training loop
    for task_idx in range(num_tasks):
        start_time = time.time()
        print("Task : " + str(task_idx))
        x_train, y_train, x_test, y_test = load_imagenet(class_order[task_idx * 2:(task_idx + 1) * 2])
        x_train, x_test = x_train.float(), x_test.float()
        # x_train, x_test = x_train.flatten(1), x_test.flatten(1) for linear network

        x_train, y_train = x_train.to(dev), y_train.to(dev)
        x_test, y_test = x_test.to(dev), y_test.to(dev)



        # Epoch loop
        for epoch_idx in range(num_epochs):  # tqdm
            example_order = np.random.permutation(examples_per_epoch)
            x_train, y_train = x_train[example_order], y_train[example_order]

            for i, start_idx in enumerate(range(0, examples_per_epoch, mini_batch_size)):
                batch_x = x_train[start_idx:start_idx + mini_batch_size]
                batch_y = y_train[start_idx:start_idx + mini_batch_size]
                # show_batch(batch_x, batch_y, num_images_to_show=10, denormalize=True)
                loss, network_output = learner.learn(x=batch_x, target=batch_y)
        weight_layer[task_idx] = net.layers[-1].weight.data
        bias_layer[task_idx] = net.layers[-1].bias.data
        #timesafe
        training_time += (time.time() - start_time)

        activations = {}
        inputs_to_activations = {}

        hooks = []
        for name, layer in net.named_modules():
            if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
                hooks.append(layer.register_forward_hook(get_activation(name)))

        #Eval 100 tasks(also the eval of current task) fehlt noch die heads die man speichert
        for t, previous_task_idx in enumerate(np.arange(max(0, task_idx - 99), task_idx + 1)):
            net.layers[-1].weight.data = weight_layer[previous_task_idx]
            net.layers[-1].bias.data = bias_layer[previous_task_idx]
            x_train, y_train, x_test, y_test = load_imagenet(class_order[previous_task_idx * 2:(previous_task_idx + 1) * 2])
            x_train, x_test = x_train.float(), x_test.float()
            x_test, y_test = x_test.to(dev), y_test.to(dev)
            prev_accuracies = torch.zeros(x_test.shape[0] // mini_batch_size, dtype=torch.float)
            for i, start_idx in enumerate(range(0, x_test.shape[0], mini_batch_size)):
                test_batch_x = x_test[start_idx:start_idx + mini_batch_size]
                test_batch_y = y_test[start_idx:start_idx + mini_batch_size]
                network_output, _ = net.predict(x=test_batch_x)
                prev_accuracies[i] = accuracy(F.softmax(network_output, dim=1), test_batch_y)
            historical_accuracies[task_idx][task_idx-previous_task_idx] = prev_accuracies.mean().item()
        #Input Distribution
        for i, start_idx in enumerate(range(0, 99, 1)):
            test_batch_x = x_test[start_idx:start_idx + 1]
            test_batch_y = y_test[start_idx:start_idx + 1]
            network_output, _ = net.predict(x=test_batch_x)

            # Process stored inputs to activation functions
            for layer_idx, (layer_name, input_data) in enumerate(inputs_to_activations.items()):
                if "fc" in layer_name:
                    input_distribution[task_idx][int(layer_name[-1])-1][0][i] = input_data.mean().item()

        # Count dormant neurons
        for t_idx, threshold in enumerate(np.arange(-20, 20, 0.01)):
            dormant_neurons[task_idx][t_idx] = average_activation_input(activations, threshold=threshold, model=net)
        # Remove hooks after counting
        for hook in hooks:
            hook.remove()
        #head reset for new task
        net.layers[-1].weight.data.zero_()
        net.layers[-1].bias.data.zero_()

    # Final save
    save_data({
        'last100_accuracies' :historical_accuracies,
        'time per task'  : training_time/num_tasks, #Training Time
        'average_activation_input': dormant_neurons.cpu(),
        'input_distribution': input_distribution.cpu(),
    }, data_file)





