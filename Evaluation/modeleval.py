import torch
import numpy as np
import matplotlib.pyplot as plt
from lop.nets.conv_net import ConvNet
import os
import pickle
import matplotlib.colors as mcolors
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
class_order_path = os.path.join(script_dir, "..", "lop", "imagenet", "class_order")
class_order_path = os.path.normpath(class_order_path)
with open(class_order_path, 'rb') as f: class_order = pickle.load(f)[3]
class_order = np.concatenate([class_order] * ((2 * 2000) // 1000 + 1))
def load_imagenet(classes=[]):
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, _class in enumerate(classes):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        class_order_path = os.path.join(script_dir, "..", "lop", "imagenet", "data","classes","")
        data_file = class_order_path + str(_class) + '.npy'
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
def normalize(d):
    norm = torch.sqrt(sum((torch.norm(p) ** 2 for p in d)))
    return [p / norm for p in d]
dev ="cpu"

# Use a fixed random seed



"""# Set device
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure the output directory exists
output_dir = "density_frames"
os.makedirs(output_dir, exist_ok=True)


# Function to plot density
def plot_density(weights_data, biases_data, task_idx, output_path):
    plt.figure(figsize=(10, 6))

    # Density plot for weights
    sns.kdeplot(weights_data, fill=True, color="blue", alpha=0.6, label="Weights")

    # Density plot for biases
    sns.kdeplot(biases_data, fill=True, color="red", alpha=0.6, label="Biases")

    plt.title(f"Density of Weights & Biases (Task {task_idx})")
    plt.xlim(-3, 3)
    plt.ylim(0, 4)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path)
    plt.close()


# Task indices
task_indices = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999, 1099, 1199, 1299, 1399, 1499, 1599, 1699, 1799,
                1899, 1999]

# Generate density plots for each task
frame_paths = []
for task_idx in task_indices:
    model = ConvNet(activation="relu")
    model.load_state_dict(
        torch.load(f'RELUmodelweights/model_weights_{task_idx}.pth', map_location=torch.device('cpu')))
    model.to(dev)

    # Collect weights and biases separately
    weights_values = []
    biases_values = []

    for param in model.named_parameters():
        values = param[1].cpu().detach().numpy().flatten()
        if "fc" in param[0]:
            if "bias" in param[0]:  # Bias values
                print("a")
                biases_values.extend(values)
            else:  # Weight values
                weights_values.extend(values)

    # Plot density
    output_path = os.path.join(output_dir, f"density_task_{task_idx}.png")
    plot_density(weights_values, biases_values, task_idx, output_path)

    frame_paths.append(output_path)

# Create GIF
frames = [Image.open(frame_path) for frame_path in frame_paths]
gif_path = "weights_bias_density.gif"

frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=500,  # Frame duration in milliseconds
    loop=0  # Infinite loop
)

print(f"GIF created at: {gif_path}")"""
"""torch.manual_seed(42)

# Assign the same function to both
random1 = torch.randn_like
random2 = torch.randn_like
for task_idx in [1599,1699,1799,1899,1999]:
    x_train, y_train, x_test, y_test = load_imagenet(class_order[task_idx * 2:(task_idx + 1) * 2])
    x_train, x_test = x_train.float(), x_test.float()
    x_test, y_test = x_test.to(dev), y_test.to(dev)

    # Define model and load weights
    model = ConvNet(activation="relu")
    model.load_state_dict(torch.load('RELUmodelweights/model_weights_'+ str(task_idx)+'.pth', map_location=torch.device('cpu')))
    model.to(dev)

    loss_fn = torch.nn.CrossEntropyLoss()

    # Generate random directions
    torch.manual_seed(42)  # Reset the random seed
    d1 = normalize([random1(p) for p in model.parameters()]) #normalize

    torch.manual_seed(93)  # Reset the random seed
    d2 = normalize([random2(p) for p in model.parameters()])

    # Define grid for alpha and beta
    alphas = np.linspace(-10000, 10000, 20)  # Double the range
    betas = np.linspace(-10000, 10000, 20)

    loss_values = np.zeros((20, 20))

    original_params = [p.clone() for p in model.parameters()]

    # Define batch size for data processing
    mini_batch_size = 64
    # Compute loss landscape
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            for name, (p, orig_p, d1_p, d2_p) in zip(model.named_parameters(),zip(model.parameters(), original_params, d1, d2)):
                if 'bias' in name[0] and 'fc' in name[0]:# or name[0].find('fc2')  or name[0].find('fc3') :
                    p.data = orig_p + alpha * d1_p + beta * d2_p
                    #print(name[0])
                    #print(p.data)
                    #print(orig_p)
                    #print("-----")
            # Compute loss over mini-batches
            total_loss = 0.0
            total_samples = 0

            for start_idx in range(0, x_test.shape[0], mini_batch_size):
                test_batch_x = x_test[start_idx:start_idx + mini_batch_size]
                test_batch_y = y_test[start_idx:start_idx + mini_batch_size]
                # Forward pass
                with torch.no_grad():
                    network_output, _ = model.predict(x=test_batch_x)
                    loss = loss_fn(network_output, test_batch_y.long())

                total_loss += loss.item() * test_batch_x.size(0)
                total_samples += test_batch_x.size(0)

            # Average the loss over the dataset
            loss_values[i, j] = total_loss / total_samples

    # Restore original parameters
    for p, orig_p in zip(model.parameters(), original_params):
        p.data = orig_p


    # Define the levels for the colormap
    levels = np.linspace(0, 5, 50)  # Levels from 0 to 5
    levels = np.append(levels, [loss_values.max()])  # Include one level for values over 5

    # Create a colormap where values above 5 are the same color
    cmap = plt.get_cmap('viridis')
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=True)

    # Plot the loss landscape
    X, Y = np.meshgrid(alphas, betas)
    plt.contourf(X, Y, loss_values, levels=levels, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.title("Loss Landscape")
    plt.xlabel("Alpha")
    plt.ylabel("Beta")
    plt.savefig(f'frame_{task_idx}.png')  # Save each frame as a PNG file
    plt.clf()  # Clear the current figure
with imageio.get_writer('loss_landscape.gif', mode='I') as writer:
    for i in [0,99,199,299,399,499,599,699,699,799,899,999,1099,1199,1299,1399,1499,1599,1699,1799,1899,1999]:
        filename = f'frame_{i}.png'
        image = imageio.imread(filename)
        writer.append_data(image)
"""
