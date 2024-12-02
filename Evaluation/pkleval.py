import pickle
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from scipy.ndimage import gaussian_filter1d

# Load data
with open('outputRELU.pkl', 'rb') as file:
    data = pickle.load(file)

# Extract the historical accuracies
print(data["time per task"])
historical_accuracies = data["last100_accuracies"]

"""# Convert PyTorch tensor to NumPy array (if needed)
if isinstance(historical_accuracies, torch.Tensor):
    historical_accuracies = historical_accuracies.numpy()

# Output directory for images
output_dir = "task_frames"
os.makedirs(output_dir, exist_ok=True)

# Generate a plot for every 50th task, averaging ±2 tasks
image_files = []
num_tasks = len(historical_accuracies)
for task_idx in range(0, num_tasks, 50):  # Every 50th task
    # Get indices for ±2 tasks around the current 50th task
    start_idx = max(0, task_idx - 10)  # Ensure we don't go below 0
    end_idx = min(num_tasks, task_idx + 10)  # Ensure we don't exceed the number of tasks

    # Calculate the average accuracy for tasks in this range
    avg_accuracies = np.mean(historical_accuracies[start_idx:end_idx], axis=0)

    # Plot the average accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(100), avg_accuracies, marker='o', label=f'Task {task_idx} ±10 Avg')
    plt.xlim(-1,100)
    plt.ylim(0,1)
    plt.title(f'Average Accuracy Around Task {task_idx}')
    plt.xlabel('x (sub-index within task)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    # Save the plot as an image
    image_file = os.path.join(output_dir, f'task_{task_idx}_avg.png')
    plt.savefig(image_file)
    plt.close()
    image_files.append(image_file)

# Create a GIF from the saved images
gif_path = "accuracy_tasks_avg_around_50th.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
    for image_file in image_files:
        writer.append_data(imageio.imread(image_file))

# Clean up images (optional)
for image_file in image_files:
    os.remove(image_file)"""




# Extract activations
activations = data['task_activations']
if isinstance(activations, torch.Tensor):
    activations = activations.numpy() # torch.zeros(num_tasks,3,3,128, 200)

plt.figure(figsize=(10, 6))


for x in range(128):
    flattened_activations = activations[1, 0, 0,x]
    flattened_activations1 = activations[0, 1, 0,x]
    sns.kdeplot(flattened_activations.flatten(),color="blue", alpha=1)
    sns.kdeplot(flattened_activations1.flatten(), color="red", alpha=1)
plt.xlim(-100,100)
plt.title("Density Distribution of Activations")
plt.xlabel("Activation Values")
plt.ylabel("Density")
plt.grid(True)
plt.show()
