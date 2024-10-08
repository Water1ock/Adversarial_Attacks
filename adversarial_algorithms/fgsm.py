import torch
import torch.nn.functional as F

def fgsm_attack(model, data_loader, epsilon):
    """
    Perform FGSM attack on the given model using the CIFAR-10 dataset.

    Args:
    - model: The neural network model on which to perform the attack.
    - data_loader: DataLoader containing the dataset (usually test dataset).
    - epsilon: The perturbation size (how much to modify the input).

    Returns:
    - original_acc: The accuracy of the model on the original dataset.
    - final_acc: The accuracy of the model after the attack.
    - adv_examples: A list of adversarial examples, their original images, and labels.
    """

    # Set the model to evaluation mode
    model.eval()

    adv_examples = []
    correct = 0
    total = 0
    original_correct = 0
    original_total = 0

    # Evaluate original accuracy first
    for data, target in data_loader:
        data, target = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        output = model(data)
        _, predicted = output.max(1, keepdim=True)
        original_correct += (predicted == target.view_as(predicted)).sum().item()
        original_total += target.size(0)

    original_acc = original_correct / float(original_total)
    print(f"Original Accuracy: {original_acc:.4f}")

    for data, target in data_loader:
        data, target = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), target.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Set requires_grad to True for the attack
        data.requires_grad = True

        # Forward pass through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # Get the initial prediction

        # Calculate the loss
        loss = F.cross_entropy(output, target)

        # Zero the gradients before backward pass
        model.zero_grad()

        # Backward pass to calculate the gradient of the loss w.r.t the input data
        loss.backward()

        # Get the sign of the gradients
        data_grad = data.grad.data

        # Create the perturbed image by adjusting the input data with epsilon * sign(gradient)
        perturbed_data = data + epsilon * data_grad.sign()

        # Clip the perturbed data to stay within the valid image range [0, 1]
        perturbed_data = torch.clamp(perturbed_data, 0, 1)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check if the perturbed image is correctly classified
        final_pred = output.max(1, keepdim=True)[1]  # Get the new prediction

        # Count correct predictions for adversarial examples
        correct += (final_pred == target.view_as(final_pred)).sum().item()

        # Save adversarial examples along with original images and their labels
        for i in range(len(target)):
            if len(adv_examples) < 5:  # Save a few examples for review
                adv_ex = perturbed_data[i].detach().cpu().numpy()
                original_image = data[i].detach().cpu().numpy()
                adv_examples.append((init_pred[i].item(), final_pred[i].item(), adv_ex, original_image, target[i].item()))

        total += target.size(0)

    # Calculate final accuracy after the attack
    final_acc = correct / float(total)
    print(f"FGSM Attack - Epsilon: {epsilon}\tTest Accuracy = {correct} / {total} = {final_acc:.4f}")

    return original_acc, final_acc, adv_examples