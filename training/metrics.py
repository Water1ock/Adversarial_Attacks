import torch

def calculate_accuracy(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients during evaluation
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Get predictions
            _, predicted = torch.max(output.data, 1)

            # Count correct predictions
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total  # Calculate accuracy
    return accuracy

def calculate_loss(output, target, criterion):
    loss = criterion(output, target)
    return loss.item()  # Return loss as a scalar

def calculate_precision(predicted, target):
    """
    Calculate the precision of the model predictions.

    Args:
    - predicted: Predicted labels from the model.
    - target: Ground truth labels.

    Returns:
    - precision: The precision score as a float.
    """
    predicted = predicted.view(-1)
    target = target.view(-1)

    true_positives = (predicted * target).sum().item()
    false_positives = (predicted * (1 - target)).sum().item()

    if true_positives + false_positives == 0:
        return 0.0
    precision = true_positives / (true_positives + false_positives)
    return precision

def calculate_recall(predicted, target):
    """
    Calculate the recall of the model predictions.

    Args:
    - predicted: Predicted labels from the model.
    - target: Ground truth labels.

    Returns:
    - recall: The recall score as a float.
    """
    predicted = predicted.view(-1)
    target = target.view(-1)

    true_positives = (predicted * target).sum().item()
    false_negatives = ((1 - predicted) * target).sum().item()

    if true_positives + false_negatives == 0:
        return 0.0
    recall = true_positives / (true_positives + false_negatives)
    return recall