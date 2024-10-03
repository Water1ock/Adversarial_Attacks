import torch
from .metrics import calculate_accuracy, calculate_precision, calculate_recall

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset.

    Args:
    - model: The neural network model to evaluate.
    - test_loader: DataLoader for the test dataset.
    - device: Device to run the model on ('cuda' or 'cpu').

    Returns:
    - test_accuracy: The accuracy of the model on the test dataset.
    - test_precision: The precision of the model on the test dataset.
    - test_recall: The recall of the model on the test dataset.
    """

    model.eval()  # Set the model to evaluation mode

    all_predicted = []
    all_targets = []

    with torch.no_grad():  # No need to track gradients during evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Get predictions
            _, predicted = torch.max(output.data, 1)

            all_predicted.append(predicted)
            all_targets.append(target)

    # Concatenate all predictions and targets for metric calculations
    all_predicted = torch.cat(all_predicted)
    all_targets = torch.cat(all_targets)

    # Calculate accuracy, precision, and recall
    test_accuracy = calculate_accuracy(model, test_loader, device)
    test_precision = calculate_precision(all_predicted, all_targets)
    test_recall = calculate_recall(all_predicted, all_targets)

    print(f'Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, '
          f'Test Recall: {test_recall:.4f}')

    return test_accuracy, test_precision, test_recall