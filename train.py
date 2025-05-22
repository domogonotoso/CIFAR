# train.py
from utils.data_utils import get_train_valid_loaders
from utils.utils import train

from models.MNIST import MnistCNN

def main():
    train_loader, valid_loader = get_train_valid_loaders(
        batch_size=64, valid_ratio=0.2, data_dir='./data'
    )

    model = MnistCNN()
    train(model, train_loader, valid_loader, num_epochs=10)

if __name__ == "__main__": # Execute itself when run this file directly.
    main()
