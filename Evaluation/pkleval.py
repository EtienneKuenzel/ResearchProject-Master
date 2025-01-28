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
    #'outputRELU.pkl',
    #'outputleakyRELU.pkl',
    #'outputPAU.pkl',
    #'outputtanh.pkl',
    #'outputPAUfreeze.pkl',
    #'outputRELU+up.pkl',
    #'outputRELU+down.pkl',
    'outputtest.pkl']
labels = ["correlation",
          'RELU',
          #'LeakyRELU',
          'PAU',
          #'Tanh',
          'PAU+freeze', "RELU+u", "RELU+d"]
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

smoothed_accuracies = [
    moving_average([accuracy[x, 0] for x in range(150)], 10)
    for accuracy in historical_accuracies]

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

# Output directory for images
output_dir = "task_frames"
os.makedirs(output_dir, exist_ok=True)

# Generate and save average accuracy plots
image_files = []
num_tasks = len(historical_accuracies[0])
step = 50  # Interval for averaging

"""for task_idx in range(0, num_tasks, step):
    avg_accuracies = [
        np.mean(accuracy[max(0, task_idx - step):min(num_tasks, task_idx + step)], axis=0)
        for accuracy in historical_accuracies
    ]

    plt.figure(figsize=(10, 6))
    for avg, label, color in zip(avg_accuracies, labels, colors):
        plt.plot(range(100), avg, marker='o', color=color, label=f'{label}: Task {task_idx}')

    plt.xlim(-1, 100)
    plt.ylim(0, 1)
    plt.title(f'Average Accuracy Around Task {task_idx}')
    plt.xlabel('Sub-index within task')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    image_file = os.path.join(output_dir, f'task_{task_idx}_avg.png')
    plt.savefig(image_file)
    plt.close()
    image_files.append(image_file)

# Create a GIF from saved images
gif_path = "accuracy_tasks_avg_around_50th.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
    for image_file in image_files:
        writer.append_data(imageio.imread(image_file))"""


frames = []
def custom_activation(data):
    # Apply the custom transformation
    transformed_data = np.where(data < -3, 0, np.maximum(0, data))
    return transformed_data
num_activation_tasks = 150  # Number of datapoints(equal to task_number/eval_every_tasks in singly_expr.py)
for i in range(num_activation_tasks):
    #i=2
    for activation, label, color in zip(activations, labels, colors):
        sns.kdeplot(activation[i, 0, 0].flatten(), fill=True, alpha=0.2, label=label)
        print(activation[i, 0, 0].flatten())
    plt.xlim(-5, 5)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    filename = f"frame_{i}.png"
    plt.savefig(filename)
    frames.append(filename)
    plt.clf()
# Create a GIF for activations
gif_activation_path = "activations.gif"
with imageio.get_writer(gif_activation_path, mode="I", fps=1) as writer:
    for frame in frames: writer.append_data(imageio.imread(frame))
