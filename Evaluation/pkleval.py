import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data





import numpy as np
import matplotlib.pyplot as plt
import pickle

def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def plot_average_ttp(file_paths, window_size=4, selected_indices=[0,1,2]):
    labels = ['Relu', "Relu+down", "Tanh",
              "PAU", "Relu+down(Convolutions locked)",
              "Relu+down(FC locked)"]
    colors = ["#045275", "#089099", "#7CCBA2", "#F0746E", "#DC3977", "#7C1D6F"]

    plt.figure(figsize=(12, 5.8))

    for i1, path in enumerate(file_paths):
        run_smoothed = []
        for run_idx in selected_indices:
            data = load_data(path + str(run_idx) + ".pkl")
            data = data[0]["last100_accuracies"]
            values = [x[0] for i, x in enumerate(data) if i % 25 == 0]
            smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
            run_smoothed.append(smoothed)
        run_smoothed = np.array(run_smoothed)
        mean_vals = np.mean(run_smoothed, axis=0)
        std_vals = np.std(run_smoothed, axis=0)
        x_pos = np.arange(0, 25 * len(mean_vals), 25)
        plt.plot(x_pos, mean_vals, label=labels[i1], color=colors[i1])
        plt.fill_between(x_pos, mean_vals - std_vals, mean_vals + std_vals,color=colors[i1], alpha=0.3)
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim(0, 5000)
    plt.ylim(0.5, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np




def plot_stability(file_paths, window_size=4, selected_indices=[0,1,2,3]):
    labels = ['Relu', "Relu+down", "Tanh", "DBP"]
    colors = ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]

    plt.figure(figsize=(12, 5.8))
    means = []

    for i1, path in enumerate(file_paths):
        mean = []
        b=[]
        for run_idx in selected_indices:
            data = load_data(path + str(run_idx) + ".pkl")
            last100 = data[0]["last100_accuracies"]
            a = [sum(x[1:]) / len(x[1:]) for i, x in enumerate(last100) if i % 25 == 0 and i > 4000]
            mean.append(sum(a) / len(a))
        means.append(np.mean(mean))

    x_labels = [f"Group {i}" for i in range(len(means))]
    barlist = plt.bar(labels, means)

    # Apply colors if number of file_paths <= number of colors
    for i, bar in enumerate(barlist):
        if i < len(colors):
            bar.set_color(colors[i])

    plt.xlabel('Configuration')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt
import torch  # Ensure torch is imported
import numpy as np

def plot_stabilityline(file_paths, window_size=4, selected_indices=[0, 1, 2, 3]):
    labels = ['Relu', "Relu+down", "Tanh", "DBP"]
    colors = ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]

    plt.figure(figsize=(12, 5.8))

    for i1, path in enumerate(file_paths):
        accumulated = []

        for run_idx in selected_indices:
            data = load_data(path + str(run_idx) + ".pkl")
            last100 = data[0]["last100_accuracies"]
            subset = last100[::25][-20:]

            average_tensor = subset.mean(dim=0)
            accumulated.append(average_tensor)

        # Average across runs
        overall_mean = torch.stack(accumulated).mean(dim=0).numpy()

        # Plot line
        plt.plot(overall_mean, label=labels[i1], color=colors[i1])

    plt.xlabel("Class Index")
    plt.ylabel("Accuracy")
    plt.title("Stability Over Final 100 Classes (Sampled Every 25)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





def timepertask(filepaths):
    selected_indices=[0]
    for path in filepaths:
        a = []
        for run_idx in selected_indices:
            data = load_data(path + str(run_idx) + ".pkl")
            a.append(data[0]['time per task'])
        print(sum(a) / len(a))
def activation(filepaths):
    data_list = []
    for file_path in filepaths:
        with open(file_path, 'rb') as file:
            data_list.append(pickle.load(file))
    activations = [data[0]['task_activations'] for data in data_list]
    labels = ["ReLU"]
    colors = ["red","blue","green", "brown", "pink"]
    for activation, label, color in zip(activations, labels, colors):
        print(len(activation))
        frames =[]
        for i in range(203):#20
            #i*=50
            plt.figure(figsize=(6, 6))  # Optional: Adjust figure size for better clarity
            data = activation[i][0, 0, 0].flatten()
            mean_value = np.mean(data.cpu().numpy())  # Convert tensor to NumPy before computing mean
            sns.kdeplot(data, fill=True, alpha=0.2, label=label, color="blue")  # KDE plot
            plt.axvline(mean_value, linestyle="dashed", color="red", alpha=0.8, linewidth=2, label=f"{label} Mean")  # Mean line
            plt.legend(prop={'weight': 'bold', 'size': 16})  # bold and larger text
            plt.xlim(-20, 20)
            plt.ylim(0, 0.5)
            plt.xticks(fontsize=12, fontweight='bold')
            plt.yticks(fontsize=12, fontweight='bold')
            plt.grid(True)
            plt.xlabel('Parameter Values', fontsize=14, fontweight='bold')
            plt.ylabel('Density', fontsize=14, fontweight='bold')
            plt.tight_layout()

            filename = f"frame_{i}.png"
            plt.savefig(filename)
            frames.append(filename)
            plt.clf()
            # Create a GIF for activations
        with imageio.get_writer(label + ".gif", mode="I", fps=10) as writer:
            for frame in frames: writer.append_data(imageio.imread(frame))
if __name__ == "__main__":
    file_paths = [
        "relu",
        "reludown",
        "tanh",
        "dbpreludown"
        #"leakyrelu2.pkl"
    ]  # Add both files
    #timepertask(file_paths)
    #plot_stability(file_paths)
    #plot_stabilityline(file_paths)
    plot_average_ttp(file_paths)
    #activation(file_paths)