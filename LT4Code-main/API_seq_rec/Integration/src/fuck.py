import matplotlib.pyplot as plt

def plot_graph_and_save(data, title, filename):
    """
    Plot a graph based on the provided data and save it as a PNG file.

    Parameters:
        data (dict): A dictionary containing the threshold, accuracy, and rejection rate.
        title (str): The title of the graph.
        filename (str): The name of the file to save the graph.
    """
    thresholds = data["threshold"]
    accuracies = data["accuracy"]
    rejection_rates = data["rejection_rate"]

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, accuracies, marker="o", label="Accuracy (%)")
    plt.plot(thresholds, rejection_rates, marker="o", label="Rejection Rate (%)")

    
    plt.xlabel("Threshold", fontsize=14)
    plt.ylabel("Percentage (%)",fontsize=14)
    plt.xticks(thresholds)
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=14)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filename)
    plt.close()


def main():
    # Data for APIシーケンス推薦
    sequence_data = {
        "threshold": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "accuracy": [57.0, 57.0, 57.1, 57.1, 57.3, 58.6, 61.7, 65.4, 69.5, 73.1],
        "rejection_rate": [85.1, 85.1, 85.1, 85.1, 85.2, 85.5, 86.3, 87.1, 88.0, 88.7]
    }

    # Data for 単一APIメソッド推薦
    method_data = {
        "threshold": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "accuracy": [62.3, 62.3, 62.4, 62.5, 62.6, 63.4, 66.1, 68.7, 71.0, 72.8],
        "rejection_rate": [52.3, 52.3, 52.4, 52.5, 52.7, 53.6, 55.7, 57.6, 59.2, 60.4]
    }

    # Plot and save graphs
    plot_graph_and_save(sequence_data, "Threshold vs Accuracy and Rejection Rate (API Method Sequence Recommendation)", "../data/sequence_recommendation.png")
    plot_graph_and_save(method_data, "Threshold vs Accuracy and Rejection Rate (Single API Method Recommendation)", "../data/method_recommendation.png")

if __name__ == "__main__":
    main()
