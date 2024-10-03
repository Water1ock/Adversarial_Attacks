import torchvision.transforms as transforms

def get_augmentations():
    """
    Returns a composition of transformations for data augmentation.

    Returns:
    - transform: A composed transformation for augmenting training images.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomCrop(32, padding=4),  # Randomly crop with padding
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
        transforms.ToTensor(),  # Convert PIL Image to tensor
    ])