import pickle
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio

# Load data
with open('outputRELU.pkl', 'rb') as file:
    data = pickle.load(file)
with open('outputleakyRELU.pkl', 'rb') as file:
    data1 = pickle.load(file)
with open('outputPAU.pkl', 'rb') as file:
    data2 = pickle.load(file)
with open('outputtanh.pkl', 'rb') as file:
    data3 = pickle.load(file)
# Extract the historical accuracies
print(data["time per task"])
print(data1["time per task"])
print(data2["time per task"])
print(data3["time per task"])

historical_accuracies = data["last100_accuracies"].numpy()
historical_accuracies1 = data1["last100_accuracies"].numpy()
historical_accuracies2 = data2["last100_accuracies"].numpy()
historical_accuracies3 = data3["last100_accuracies"].numpy()
# Extract activations
activations = data['task_activations'].numpy() # torch.zeros(num_tasks,3,3,128, 200)
activations1 = data1['task_activations'].numpy() # torch.zeros(num_tasks,3,3,128, 200)
activations2 = data2['task_activations'].numpy() # torch.zeros(num_tasks,3,3,128, 200)
activations3 = data3['task_activations'].numpy() # torch.zeros(num_tasks,3,3,128, 200)

# Plot each dataset with a unique color
relu, leakrelu, pau, sig = [], [],[],[]
for x in range(2000):
    relu.append(historical_accuracies[x, 0])
    leakrelu.append(historical_accuracies1[x, 0])
    pau.append(historical_accuracies2[x, 0])
    sig.append(historical_accuracies3[x, 0])

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Apply moving average with a window size of 50 (adjust as needed)
window_size = 50
relu_smoothed = moving_average(relu, window_size)
leakrelu_smoothed = moving_average(leakrelu, window_size)
pau_smoothed = moving_average(pau, window_size)
sig_smoothed = moving_average(sig, window_size)

# Plot original and smoothed data
plt.figure(figsize=(12, 6))
plt.plot(range(len(relu_smoothed)), relu_smoothed, linestyle='-', color='blue', label='RELU')
plt.plot(range(len(leakrelu_smoothed)), leakrelu_smoothed, linestyle='-', color='red', label='LeakyRelu')
plt.plot(range(len(pau_smoothed)), pau_smoothed, linestyle='-', color='green', label='PAU')
plt.plot(range(len(sig_smoothed)), sig_smoothed, linestyle='-', color='purple', label='Tanh')

# Add labels and title
plt.xlabel("Index")
plt.ylabel("Accuracy")
plt.title("Historical Accuracies (Last 300)")
plt.ylim(0.5, 1)  # Accuracy values are between 0 and 1
plt.xticks([0,100, 200,300, 400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000])

plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# Output directory for images
output_dir = "task_frames"
os.makedirs(output_dir, exist_ok=True)

# Generate a plot for every 50th task, averaging ±2 tasks
image_files = []
num_tasks = len(historical_accuracies)
for task_idx in range(0, num_tasks, 50):  # Every 50th task
    # Get indices for ±2 tasks around the current 50th task
    start_idx = max(0, task_idx - 50)  # Ensure we don't go below 0
    end_idx = min(num_tasks, task_idx + 50)  # Ensure we don't exceed the number of tasks

    # Calculate the average accuracy for tasks in this range
    avg_accuracies = np.mean(historical_accuracies[start_idx:end_idx], axis=0)
    avg_accuracies1 = np.mean(historical_accuracies1[start_idx:end_idx], axis=0)
    avg_accuracies2 = np.mean(historical_accuracies2[start_idx:end_idx], axis=0)
    avg_accuracies3 = np.mean(historical_accuracies3[start_idx:end_idx], axis=0)

    # Plot the average accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(100), avg_accuracies, marker='o',color="blue", label=f'RELU Task {task_idx}')
    plt.plot(range(100), avg_accuracies1, marker='o',color="red", label=f'LeakyRELU: Task {task_idx}')
    plt.plot(range(100), avg_accuracies2, marker='o',color="green", label=f'PAU: Task {task_idx}')
    plt.plot(range(100), avg_accuracies3, marker='o',color="purple", label=f'Tanh: Task {task_idx}')
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







frames = []  # To store filenames for the GIF
for i in range(20) :
    plt.figure(figsize=(10, 6))

    flattened_activations = activations[i, 0, 0]
    flattened_activations1 = activations1[i, 0, 0]
    flattened_activations2 = activations2[i, 0, 0]
    flattened_activations3 = activations3[i, 0, 0]
    sns.kdeplot(flattened_activations.flatten(), fill=True, color="blue", alpha=0.2, label="RELU")
    sns.kdeplot(flattened_activations1.flatten(), fill=True, color="red", alpha=0.2, label="leakyRELU")
    sns.kdeplot(flattened_activations2.flatten(), fill=True, color="green", alpha=0.2, label="PAU")
    sns.kdeplot(flattened_activations3.flatten(), fill=True, color="purple", alpha=0.2, label="Tanh")

    plt.title(f"Density Distribution of Activations (Task: {i*100})")
    plt.xlabel("Activation Values")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.xlim(-100, 100)
    plt.ylim(0, 0.2)
    # Save the plot to a file
    filename = f"frame_{i}.png"
    plt.savefig(filename)
    frames.append(filename)
    plt.close()  # Close the figure to save memory

# Create a GIF
with imageio.get_writer("activations.gif", mode="I", fps=1) as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

