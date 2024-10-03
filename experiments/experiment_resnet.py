import torch
import argparse
from models.resnet import ResNet18
from data.cifar10_loader import get_cifar10_loaders
from training.trainer import train_model
from adversarial_algorithms.fgsm import fgsm_attack
from utils.visualization import visualize_images, plot_accuracy_comparison

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)

    # Initialize model
    model = ResNet18()

    # Train the model
    train_model(model, train_loader, test_loader, num_epochs=args.num_epochs, learning_rate=args.learning_rate, device=device)

    # Perform FGSM attack
    original_acc, final_acc, adv_examples = fgsm_attack(model, test_loader, epsilon=args.epsilon)

    # Visualize original and adversarial examples
    original_images = [adv[3] for adv in adv_examples]  # Original images from adv examples
    adversarial_images = [adv[2] for adv in adv_examples]  # Adversarial images from adv examples
    labels = [adv[4] for adv in adv_examples]  # True labels from adv examples
    preds = [adv[1] for adv in adv_examples]  # Predicted labels from adv examples

    visualize_images("ResNet18", original_images, adversarial_images, labels, preds)

    # Plot accuracy comparison
    plot_accuracy_comparison("ResNet18", original_acc, final_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment with ResNet on CIFAR-10')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for DataLoader')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon for FGSM attack')

    args = parser.parse_args()
    main(args)