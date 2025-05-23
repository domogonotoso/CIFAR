import torch
import torch.nn as nn
from utils.plot_utils import plot_training
import copy

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        if current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = current_score
            self.counter = 0
        return False


def evaluate(model, data_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc


def train(model, train_loader, val_loader, num_epochs=20, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accuracies = []
    val_accuracies = []

    best_val_acc = 0.0
    best_model_state = None
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # clear previous gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() # update weights

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        val_acc = evaluate(model, val_loader, device=device)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}  Train Acc: {train_acc:.2f}%  Val Acc: {val_acc:.2f}%")

        # The higher val_acc, the better.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            best_model_state = copy.deepcopy(model.state_dict()) # Fix parameters of best model.

        # Early stopping
        if early_stopping.step(val_acc):
            print(f"Early stopping at epoch {epoch+1}")
            break

    # save graph
    plot_training(train_accuracies, val_accuracies, model.__class__.__name__)

    #Save best moel
    if best_model_state:
        torch.save(best_model_state, "model_best.pth")
        print(f"Best model saved to model_best.pth (Val Acc: {best_val_acc:.2f}%)")

    # Save last model
    torch.save(model.state_dict(), "model_last.pth")
    print(f"Last model saved to model.pth (Val Acc: {val_acc:.2f}%)")
