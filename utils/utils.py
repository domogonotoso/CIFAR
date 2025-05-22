import torch
import torch.nn as nn

def train(model, train_loader, val_loader, num_epochs=10, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # clear previous gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() # update weights

            running_loss += loss.item() * inputs.size(0) # inputs.size(0) == batch size

        avg_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device=device)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} Validation Accuracy: {val_acc:.2f}%")


    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")


def evaluate(model, data_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(): # no need to track graients
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc
