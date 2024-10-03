import matplotlib.pyplot as plt
import numpy as np

def visualize_images(original_images, adversarial_images, labels, preds, n=5):
    """
    Visualize original and adversarial images side by side.

    Args:
    - original_images: List of original images.
    - adversarial_images: List of adversarial images.
    - labels: List of true labels for the images.
    - preds: List of predicted labels for adversarial images.
    - n: Number of images to visualize.
    """
    plt.figure(figsize=(12, 6))

    for i in range(n):
        # Display original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(np.transpose(original_images[i], (1, 2, 0)))
        plt.title(f"Original\nLabel: {labels[i]}")
        plt.axis('off')

        # Display adversarial images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(np.transpose(adversarial_images[i], (1, 2, 0)))
        plt.title(f"Adversarial\nPred: {preds[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_accuracy_comparison(original_acc, final_acc):
    """
    Plot a bar chart to compare original and adversarial accuracy.

    Args:
    - original_acc: Original accuracy.
    - final_acc: Final accuracy after FGSM attack.
    """
    labels = ['Original', 'Adversarial']
    accuracies = [original_acc, final_acc]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    bars = ax.bar(x, accuracies, width, label='Accuracy', color=['blue', 'red'])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.ylim(0, 1)  # Set y-axis limits
    ax.bar_label(bars)

    plt.tight_layout()
    plt.show()

def prepare_and_visualize(adv_examples):
    """
    Prepare data for visualization and call the visualization function.

    Args:
    - adv_examples: List of tuples containing (initial_pred, final_pred, adv_ex, orig_img, target).
    """
    original_images = []
    adversarial_images = []
    labels = []
    preds = []

    for init_pred, final_pred, adv_ex, orig_img, target in adv_examples:
        original_images.append(orig_img)
        adversarial_images.append(adv_ex)
        labels.append(target)
        preds.append(final_pred)

    # Call the visualization function
    visualize_images(original_images, adversarial_images, labels, preds)