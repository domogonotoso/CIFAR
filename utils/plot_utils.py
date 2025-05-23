import matplotlib.pyplot as plt

def plot_training(train_accuracies, val_accuracies,model_name, save_path="results/train&valid_plot.png"):
    epochs = range(1, len(train_accuracies)+1)
    
    plt.figure()
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title(f"Training Progress ({model_name})")
    plt.savefig(save_path)
    plt.close()
