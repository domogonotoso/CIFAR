import matplotlib.pyplot as plt

def plot_training(train_losses, val_accuracies, save_path="results/train_plot.png"):
    epochs = range(1, len(train_losses)+1)
    
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Training Progress")
    plt.savefig(save_path)
    plt.close()
