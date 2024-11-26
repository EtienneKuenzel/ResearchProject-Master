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
activations = data['task_activations']  # Update the key if necessary
# Convert to numpy if it's a PyTorch tensor
if isinstance(activations, torch.Tensor):
    activations = activations.numpy()

thresholds = np.arange(-20, 20, 0.01)

# Total neurons count
# Temporary folder to save frames
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

# Create frames for each task
frame_paths = []
for task_idx in range(0, 2001,10):  # Tasks 1 to 2000
    # Prepare data for the current task
    task_counts = []
    task_counts1 = []
    task_counts2 = []
    for t_idx, threshold in enumerate(thresholds):
        task_counts.append(activations[task_idx - 1][0][0][t_idx]/128)
        task_counts1.append(activations[task_idx - 1][0][1][t_idx] / 128)
        task_counts2.append(activations[task_idx - 1][0][2][t_idx] / 2)
    #task_counts = gaussian_filter1d(np.gradient(task_counts, thresholds), sigma=20)
    #task_counts1 = gaussian_filter1d(np.gradient(task_counts1, thresholds), sigma=20)
    #task_counts2 = gaussian_filter1d(np.gradient(task_counts2, thresholds), sigma=20)
    # Create the plot
    plt.figure(figsize=(10, 7))

    plt.plot(thresholds, task_counts, label=f'FC1 {task_idx}', linestyle='--')
    plt.plot(thresholds, task_counts1, label=f'FC2 {task_idx}', linestyle='--')
    plt.plot(thresholds, task_counts2, label=f'FC3 {task_idx}', linestyle='--')
    plt.ylim(-0.05, 1.05)
    plt.xlim(-20.05, 20.05)
    plt.xlabel('Threshold')
    plt.ylabel('Dormant Neurons')
    plt.title(f'Dormant Neurons vs Threshold (Task {task_idx})')
    plt.legend(title='Tasks')
    plt.grid(True)

    # Save the frame
    frame_path = os.path.join(output_folder, f"frame_{task_idx:04d}.png")
    plt.savefig(frame_path)
    plt.close()
    frame_paths.append(frame_path)


# Create GIF from the saved frames
gif_path = "dormant_neurons.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
    for frame_path in frame_paths:
        writer.append_data(imageio.imread(frame_path))

# Cleanup frames (optional)
for frame_path in frame_paths:
    os.remove(frame_path)

print(f"GIF created at {gif_path}")

