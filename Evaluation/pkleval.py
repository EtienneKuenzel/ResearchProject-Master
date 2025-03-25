import pickle
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from scipy.ndimage import gaussian_filter1d

# File paths
file_paths = [
    'outputRELUfull.pkl',
    'outputleakyRELU.pkl',
    #'outputRELU+up.pkl',
    'outputRELU+down.pkl',
    #'outputPAUfreeze.pkl',
    #'outputPAU.pkl',
    #'outputtanh.pkl',
    #'outputewcpau.pkl',
    #'outputewc0.1.pkl',
    #'outputRELU+cor.pkl',
    'outputc.pkl'
]
    #'outputtent.pkl']
#file_paths=["outputewc.pkl"]#,"outputewc1.pkl","outputtest.pkl"]
labels = ["Relu",
          "LeakyRelu",
          'Relu+down',
        #'Relu+Correlation0.95',
        'Relu+Correlation0.95+LayerNorm',
        'EWC0.5',
        'EWCPAU',
          'Tent',
          'LeakyRELU',
          "RELU+u",
          "RELU+d"
          'Pau+freeze',
          'Pau',
          'Tanh',
          'Tent']
colors = ['blue', 'red', 'green', 'purple', 'pink', "brown", "grey", "orange"]

data_list = []
for file_path in file_paths:
    with open(file_path, 'rb') as file:
        data_list.append(pickle.load(file))

# Extract historical accuracies
historical_accuracies = [data["last100_accuracies"].numpy() for data in data_list]
activations = [data['task_activations'].numpy() for data in data_list]

# Plot historical accuracies with smoothing
def moving_average(data, window_size):return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def pad_with_default(array, length, default=0.5):
    """Pad the array with the default value to reach the desired length."""
    if len(array) < length:
        return np.pad(array, (0, length - len(array)), constant_values=default)
    return array[:length]

# Example with historical_accuracies
smoothed_accuracies = [
    moving_average(
        pad_with_default([accuracy[x, 0] for x in range(accuracy.shape[0])], 2000),50) for accuracy in historical_accuracies
]

plt.figure(figsize=(12, 6))
for smoothed, label, color in zip(smoothed_accuracies, labels, colors):
    plt.plot(range(len(smoothed)), smoothed, linestyle='-', color=color, label=label)

plt.xlabel("Index")
plt.ylabel("Accuracy")
plt.title("Historical Accuracies (Smoothed)")
plt.ylim(0.5, 1)
plt.legend()
plt.grid(True)
plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt

image_files = []
output_dir = "task_frames"
num_tasks = len(historical_accuracies[0])
step = 50  # Interval for averaging

# Generate per-task average accuracy plots
for task_idx in range(0, num_tasks, step):
    avg_accuracies = [
        np.mean(accuracy[max(0, task_idx - step):min(num_tasks, task_idx + step)], axis=0)
        for accuracy in historical_accuracies
    ]

    plt.figure(figsize=(10, 6))
    for avg, label, color in zip(avg_accuracies, labels, colors):
        plt.plot(range(100), avg, marker='o', color=color, label=f'{label}: Task {task_idx}')

    plt.xlim(-1, 9)
    plt.ylim(0.45, 1)
    plt.title(f'Average Accuracy Around Task {task_idx}')
    plt.xlabel('Sub-index within task')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    image_file = os.path.join(output_dir, f'task_{task_idx}_avg.png')
    plt.savefig(image_file)
    plt.close()
    image_files.append(image_file)

# Compute the average accuracy over the first 55 tasks
x = 1000
if num_tasks >= 10:
    avg_accuracies_55 = [np.mean(accuracy[:x], axis=0) for accuracy in historical_accuracies]

    # Generate and save the new plot
    plt.figure(figsize=(10, 6))
    for avg, label, color in zip(avg_accuracies_55, labels, colors):
        plt.plot(range(100), avg, marker='o', color=color, label=f'{label}')

    plt.xlim(-1, 9)
    plt.ylim(0.45, 1)
    plt.title('Overall Average Accuracy')
    plt.xlabel('Sub-index within task')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    image_file_55 = os.path.join(output_dir, 'overall_avg_accuracy_first_55_tasks.png')
    plt.savefig(image_file_55)
    plt.close()
    print(f"Saved overall average accuracy plot for the first 55 tasks to {image_file_55}")


frames = []
def custom_activation(data):
    # Apply the custom transformation
    transformed_data = np.where(data < -3, 0, np.maximum(0, data))
    return transformed_data
for i in range(10):#20
    i*=100
    plt.figure(figsize=(8, 6))  # Optional: Adjust figure size for better clarity

    for activation, label, color in zip(activations, labels, colors):
        data = activation[i, 0, 0].flatten()
        mean_value = np.mean(data)  # Compute the mean
        sns.kdeplot(data, fill=True, alpha=0.2, label=label, color=color)  # KDE plot
        plt.axvline(mean_value, linestyle="dashed", color="red", alpha=0.8, linewidth=2, label=f"{label} Mean")  # Mean line

    plt.xlim(-100, 100)
    plt.ylim(0, 0.5)
    plt.grid(True)
    plt.legend()
    plt.title("Task : " + str(i))
    filename = f"frame_{i}.png"
    plt.savefig(filename)
    frames.append(filename)
    plt.clf()
# Create a GIF for activations
gif_activation_path = "activations.gif"
with imageio.get_writer(gif_activation_path, mode="I", fps=1) as writer:
    for frame in frames: writer.append_data(imageio.imread(frame))
