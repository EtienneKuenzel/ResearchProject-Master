import torch
import pickle
from tqdm import tqdm
from lop.algos.bp import Backprop, EWC_Policy, ConvCBP, VAE
from lop.nets.conv_net import ConvNet_PAU, ConvNet_TENT, ConvNet
from torch.nn.functional import softmax
from lop.nets.linear import MyLinear
from lop.utils.miscellaneous import nll_accuracy as accuracy
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import time as time
import os
import torch.nn.init as init
import matplotlib.pyplot as plt
import torchvision


def rescale(img: torch.Tensor) -> torch.Tensor:
    """Efficiently normalize image tensor to [0, 1] for CPU."""
    if img.requires_grad:
        img = img.detach()

    min_val = torch.amin(img)
    max_val = torch.amax(img)
    scale = max_val - min_val
    if scale < 1e-5:
        return img - min_val  # All values are nearly identical
    return (img - min_val) / scale

def show_images(images, nrow=30):
    images = images.cpu().detach()
    images = images[:nrow * nrow]

    # Rescale each image individually
    images = torch.stack([rescale(img) for img in images])

    # Make grid
    grid_img = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    np_img = grid_img.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(np_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Function to display a batch of images
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
def average_activation_input(activations, layer):
    activation_inputs = []
    for layer_name, act in activations.items():
        if layer in layer_name:
            raw_input = act.detach()  # Detach to avoid tracking gradients (if needed)
            for neuron_input in raw_input.permute(1, 0):  # Iterate over neurons (assuming N x M tensor)
                activation_inputs.append(neuron_input.tolist())  # Append raw input values (as list)
    return activation_inputs
def load_imagenet(classes=[]):
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, _class in enumerate(classes):
        data_file = "D:/ResearchProject-Master/lop/imagenet/data/classes/" + str(_class) + ".npy"#'data/classes/' + str(_class) + '.npy'
        new_x = np.load(data_file)
        x_train.append(new_x[:600])
        x_test.append(new_x[600:])
        y_train.append(np.array([idx] * 600))
        y_test.append(np.array([idx] * 100))
    x_train = torch.tensor(np.concatenate(x_train))
    y_train = torch.from_numpy(np.concatenate(y_train))
    x_test = torch.tensor(np.concatenate(x_test))
    y_test = torch.from_numpy(np.concatenate(y_test))
    return x_train, y_train, x_test, y_test


def save_data(data, data_file):
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            existing_data = pickle.load(f)
    else:
        existing_data = []
    existing_data.append(data)
    with open(data_file, 'wb') as f:
        pickle.dump(existing_data, f)
def custom_activation(x):
    return np.where(x > -3, np.maximum(0, x), -x - 3)
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div
if __name__ == '__main__':
    num_tasks = 5100
    mini_batch_size = 100
    num_epochs =  250
    eval_every_tasks = 25
    runs = 5
    dgr = True
    # Device setup
    dev = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    for run in range(runs):
        data_file = "dgr" + str(run) + ".pkl"
        # Initialize network
        net = ConvNet(activation="relu")
        # Initialize learner
        learner = Backprop(
            net=net,
            step_size=0.01,
            weight_decay=0,
            device=dev,
            momentum=0.9)
        # Load class order
        with open('class_order', 'rb') as f:class_order = pickle.load(f)[run]
        class_order = np.concatenate([class_order] * ((2 * num_tasks) // 1000 + 1))

        # Initialize accuracy tracking
        task_activations = torch.zeros(int(num_tasks/eval_every_tasks)+1,3,3,128, 200)#numtasks, 3=layer, 3=CurrentTask+OOD(Next Task)+Adveserial Attack,100=Datapoints
        historical_accuracies = torch.zeros(num_tasks, 10)
        training_time = 0
        weight_layer = torch.zeros((num_tasks, 2, 128))
        bias_layer = torch.zeros(num_tasks, 2)
        saliency_maps = torch.zeros(num_tasks, 3, 128, 128)  # Example: num_tasks, 3 channels, 128x128 input size

        # Training loop
        for task_idx in range(num_tasks):
            start_time = time.time()
            print("Run : " + str(run) + " Task : " + str(task_idx))
            x_train, y_train, x_test, y_test = load_imagenet(class_order[task_idx * 2:(task_idx + 1) * 2])
            x_train, x_test = x_train.float(), x_test.float()
            x_train, y_train = x_train.to(dev), y_train.to(dev)
            x_test, y_test = x_test.to(dev), y_test.to(dev)
            # If not the first task, generate replay data from previous tasks
            if task_idx > 0 and dgr:
                x_train = x_train[:len(x_train) // 2]
                y_train = y_train[:len(y_train) // 2]
                x_test = x_test[:len(x_test) // 2]
                y_test = y_test[:len(y_test) // 2]
                with torch.no_grad():
                    gen_x, gen_y = vae.sample(600, device=dev)
                gen_x, gen_y = gen_x.to(dev), gen_y.to(dev)
                x_train = torch.cat([x_train, gen_x], dim=0)
                y_train = torch.cat([y_train, gen_y], dim=0)
            if dgr:
                vae = VAE(z_dim=128).to(dev)
                optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

            for epoch_idx in range(num_epochs):
                print(epoch_idx)
                example_order = torch.randperm(x_train.size(0))
                x_train, y_train = x_train[example_order], y_train[example_order]
                #show_images(vae.sample(600, device=dev)[0])
                for i, start_idx in enumerate(range(0, 1200, mini_batch_size)):
                    batch_x = x_train[start_idx:start_idx + mini_batch_size]
                    batch_y = y_train[start_idx:start_idx + mini_batch_size]
                    if dgr:
                        optimizer.zero_grad()
                        recon_x, mu, logvar = vae(batch_x, batch_y)
                        loss = vae_loss(recon_x, rescale(batch_x), mu, logvar)
                        loss.backward()
                        optimizer.step()
                    loss, network_output = learner.learn(x=batch_x, target=batch_y,task=task_idx, decrease=0)#braucht 90% der zeitläuft langsamer probably wegen dem decrease
            weight_layer[task_idx] = net.layers[-1].weight.data
            bias_layer[task_idx] = net.layers[-1].bias.data
            training_time += (time.time() - start_time)
            #Eval 100 tasks
            if task_idx%eval_every_tasks ==0:
                # Current Task Activations
                activations = {}
                inputs_to_activations = {}
                hooks = []
                for name, layer in net.named_modules():
                    if isinstance(layer, (torch.nn.Linear)): hooks.append(layer.register_forward_hook(get_activation(name)))
                x_train, y_train, x_test, y_test = load_imagenet(class_order[task_idx * 2:(task_idx + 1) * 2])
                x_train, x_test = x_train.float(), x_test.float()
                x_test, y_test = x_test.to(dev), y_test.to(dev)
                for i, start_idx in enumerate(range(0, x_test.shape[0], mini_batch_size)):
                    test_batch_x = x_train[start_idx:start_idx + mini_batch_size]
                    test_batch_y = y_train[start_idx:start_idx + mini_batch_size]
                    network_output, _ = net.predict(x=test_batch_x)

                for layer in ["fc1", "fc2"]:
                    task_activations[int(task_idx/eval_every_tasks)][0][int(layer[-1]) - 1] = torch.tensor(average_activation_input(activations, layer=layer), dtype=torch.float32)
                for hook in hooks: hook.remove()
                #learner.prune_merge_neurons(task_activations, task_idx) #Correlation Algorithm
                #Stability/Plasticity Eval
                for t, previous_task_idx in enumerate(np.arange(max(0, task_idx - 9), task_idx + 1)):
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
            # head reset for new task
            net.layers[-1].weight.data.zero_()
            net.layers[-1].bias.data.zero_()
        # Final save
        save_data({
            'last100_accuracies' :historical_accuracies.cpu(),
            'time per task'  : training_time/num_tasks, #Training Time
            'task_activations': task_activations.cpu(),
        }, data_file)

