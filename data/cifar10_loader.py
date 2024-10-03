import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from .augmentations import get_augmentations  # Import augmentations from augmentations.py

def get_cifar10_loaders(batch_size=64,
                        train_normalize=transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        test_normalize=transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))):
    """
    Load the CIFAR-10 dataset and apply transformations.

    Parameters:
    - batch_size: The number of samples per batch.
    - train_normalize: A normalization transformation for training images (default: standard normalization).
    - test_normalize: A normalization transformation for testing images (default: standard normalization).

    Returns:
    - trainloader: DataLoader for the training dataset.
    - testloader: DataLoader for the test dataset.
    """

    # Apply augmentations for the training set
    train_transform = transforms.Compose([
        *get_augmentations(),  # Unpack augmentations into the compose
        train_normalize
    ])

    # Define transformations for the test set
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        test_normalize
    ])

    # Load the CIFAR-10 training dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

    # Load the CIFAR-10 test dataset
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Create DataLoader for the training dataset
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Create DataLoader for the test dataset
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader