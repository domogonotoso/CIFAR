# test.py

import torch
from models.MNIST import MnistCNN
from utils.data_utils import get_test_loader
from utils.utils import evaluate

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = MnistCNN()
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)


    test_loader = get_test_loader(batch_size=64)

    test_acc = evaluate(model, test_loader, device=device)
    print(f"✅ Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
