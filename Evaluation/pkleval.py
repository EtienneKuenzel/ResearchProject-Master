import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data





def plot_average_ttp(file_paths, window_size=1, selected_indices=[0,1,2,3,4]):
    all_ttp_values = []
    x_pos = []
    labels = ['Relu',"Relu+down+decrease 0.05","Relu+down+decrease 0.10", "Relu+down+decrease 0.15", "Relu+down(Convolutions locked)", "Relu+down(FC locked)"]
    colors = ['r', 'b', 'g', 'brown', 'pink', "purple", "r"]
    plt.figure(figsize=(12, 5.8))

    for i1, path in enumerate(file_paths):
        data = load_data(path)["last100_accuracies"].numpy()
        for i2, x in enumerate(data):
            if i2 % 500 == 0:
                x_pos.append(i2)
                all_ttp_values.append(x[0])
        plt.plot(x_pos, all_ttp_values, label=labels[i1])
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim(-10,5000)
    plt.ylim(0,1)
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def compute_average_ttp(data, selected_indices, num_segments=8, segment_size=10):
    ttp_regain_values = np.array([data['last100_accuracies'][0].numpy()])
    i = 0
    a = []
    for x in ttp_regain_values[0]:
        if x[0] < 10 and  len(a)<20:
            a.append(i)
        i+=1
    all_averages = [np.mean([ttp_regain_values[0][x]], axis=0) for x in a]
    return np.mean(all_averages, axis=0)




def plot_average_ttpregain(file_paths):
    plt.figure(figsize=(12, 5))  # Increase DPI for sharpness
    labels = ["Relu", "Relu+down", "Relu+down+\nswap", "Relu+down+\ndecrease", "Relu+down+\ndecrease+swap", "a", "b"]
    colors = ['r', 'b', 'g', 'brown', 'pink', "purple", "r"]

    selected_indices = [0]  # Indices to use
    means = []
    stds = []
    valid_labels = []
    valid_colors = []

    for file_path, label, color in zip(file_paths, labels, colors):
        try:
            data = load_data(file_path)
            final_average = compute_average_ttp(data, selected_indices)
            mean_val = final_average.mean()
            std_val = final_average.std()
            print(f"{label}: Mean={mean_val}, Std={std_val}")
            means.append(mean_val/9)
            stds.append(std_val/9)
            valid_labels.append(label)
            valid_colors.append(color)
        except Exception as e:
            print(f"Skipping {label} due to error: {e}")


    if len(means) == len(valid_labels) == len(stds) == len(valid_colors):
        plt.bar(valid_labels, means, yerr=stds, capsize=8, color=valid_colors, alpha=0.85, edgecolor='black', linewidth=1.2)
        plt.ylabel('Epochs to Relearn the older tasks', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.7)
        plt.show()
    else:
        print("Error: Mismatch in data lengths, skipping plot.")


def timepertask(filepaths):
    for path in filepaths:
        data = load_data(path)
        print(data['time per task'])
def activation(filepaths):
    data_list = []
    for file_path in filepaths:
        with open(file_path, 'rb') as file:
            data_list.append(pickle.load(file))
    activations = [data['task_activations'] for data in data_list]
    labels = ["Relu"]
    colors = ["red","blue","green", "brown", "pink"]
    for activation, label, color in zip(activations, labels, colors):
        print(len(activation))
        frames =[]
        for i in range(11):#20
            #i*=50
            plt.figure(figsize=(8, 6))  # Optional: Adjust figure size for better clarity
            data = activation[i][0, 0, 0].flatten()
            mean_value = np.mean(data.cpu().numpy())  # Convert tensor to NumPy before computing mean
            sns.kdeplot(data, fill=True, alpha=0.2, label=label, color="blue")  # KDE plot
            plt.axvline(mean_value, linestyle="dashed", color="red", alpha=0.8, linewidth=2, label=f"{label} Mean")  # Mean line
            plt.xlim(-20, 20)
            plt.ylim(0, 0.5)
            plt.grid(True)
            plt.legend()
            plt.title("Task : " + str(i*10))
            plt.tight_layout()
            plt.xlabel('Parameter Values')
            plt.ylabel('Density')
            filename = f"frame_{i}.png"
            plt.savefig(filename)
            frames.append(filename)
            plt.clf()
            # Create a GIF for activations
        with imageio.get_writer(label + ".gif", mode="I", fps=10) as writer:
            for frame in frames: writer.append_data(imageio.imread(frame))
if __name__ == "__main__":
    file_paths = [
        "relu.pkl"
    ]  # Add both files
    #plot_average_ttp(file_paths)
    #timepertask(file_paths)
    activation(file_paths)