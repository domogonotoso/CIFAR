# test.py
import torch
from models.MNIST import MnistCNN
from models.Resnet_18 import ResNet18
from utils.data_utils import get_test_loader
from utils.utils import evaluate

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    best_model = ResNet18()
    best_model.load_state_dict(torch.load("model_best.pth", map_location=device))
    best_model.to(device)

    last_model = ResNet18()
    last_model.load_state_dict(torch.load("model_last.pth", map_location=device))
    last_model.to(device)


    test_loader = get_test_loader(batch_size=64)

    best_test_acc = evaluate(best_model, test_loader, device=device)
    print(f"✅ Best model Test Accuracy: {best_test_acc:.2f}%")

    last_test_acc = evaluate(last_model, test_loader, device=device)
    print(f"✅ Last model Test Accuracy: {last_test_acc:.2f}%")


if __name__ == "__main__":
    main()
