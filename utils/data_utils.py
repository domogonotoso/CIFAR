import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_train_valid_loaders(batch_size=64, valid_ratio=0.2, data_dir='./data'):
    """
    Returns train and validation dataloaders for CIFAR-10.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010))  # Normalization
    ])
    
    full_train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    
    total_size = len(full_train_dataset)
    valid_size = int(total_size * valid_ratio)
    train_size = total_size - valid_size

    train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

def get_test_loader(batch_size=64, data_dir='./data'):
    """
    Returns test dataloader for CIFAR-10.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010))
    ])
    
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
