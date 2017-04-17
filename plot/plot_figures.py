import sys
import numpy as np
import matplotlib.pyplot as plt
import re

files = {
    "dense" : "data/final/nonsparsify_14c3xlarge_data_out",
    "sparsity=.9" : "data/final/sparsify=90_14c3xlarge_data_out",
}

def extract_values(filename):
    f = open(filename, "r")
    values = []
    for line in f:
        matches = re.findall("Epoch: ([0-9\.]+), Time: ([0-9\.]+), Accuracy: ([0-9\.]+), Loss: ([0-9\.]+)", line)
        assert(len(matches) <= 1)
        if len(matches) == 1:
            values.append([float(x) for x in matches[0]])
    return values

def extract_epochs(values):
    return [x[0] for x in values]

def extract_times(values):
    return [x[1] for x in values]

def extract_accuracies(values):
    return [x[2] for x in values]

def extract_losses(values):
    return [x[3] for x in values]

def get_number_of_data_points(values):
    return len(extract_times(values))

def time_to_target_accuracy(data, target):
    t = extract_times(data)
    accs = extract_accuracies(data)
    for a, b in zip(t, accs):
        if b >= target:
            return a
    assert(0)

def plot_time_until_accuracy(data, target_acc=.995):
    methods = []
    values = []
    plt.cla()
    for method_name, data_values in data.items():
        methods.append(method_name)
        values.append(time_to_target_accuracy(data_values, target_acc))

    ind = np.arange(len(values))
    width = .8
    plt.bar(ind, values, width=width)
    plt.xticks(ind + width / 2, methods)
    plt.xlabel("Method")
    plt.title("Seconds to reach accuracy %f" % target_acc)
    plt.ylabel("Seconds to reach accuracy %f" % target_acc)
    plt.savefig("SparsifyTimeUntilAccuracy.png")

def plot_time_vs_accuracy(data):
    plt.cla()
    for method_name, data_values in data.items():
        times = extract_times(data_values)
        accuracies = extract_accuracies(data_values)
        plt.plot(times, accuracies, label=method_name)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Training Accuracy (%)")
    plt.legend(loc="lower right")
    plt.title("Time vs Accuracy")
    plt.savefig("SparsifyTimeVsAccuracy.png")

def plot_time_vs_loss(data):
    plt.cla()
    for method_name, data_values in data.items():
        times = extract_times(data_values)
        losses = extract_losses(data_values)
        plt.plot(times, losses, label=method_name)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Training Loss")
    plt.legend(loc="upper left")
    plt.title("Time vs Loss")
    plt.savefig("SparsifyTimeVsLoss.png")

def plot_time_vs_epoch(data):
    plt.cla()
    for method_name, data_values in data.items():
        times = extract_times(data_values)
        epochs = extract_epochs(data_values)
        plt.plot(times, epochs, label=method_name)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Number of full passes of data")
    plt.legend(loc="upper left")
    plt.title("Time vs Passes of Data")
    plt.savefig("SparsifyTimeVsEpoch.png")

data_values = {k:extract_values(v) for k,v in files.items()}
plot_time_until_accuracy(data_values)
plot_time_vs_loss(data_values)
plot_time_vs_accuracy(data_values)
plot_time_vs_epoch(data_values)
