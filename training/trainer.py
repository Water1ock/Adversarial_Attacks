import torch
import torch.optim as optim
import torch.nn as nn
from .metrics import calculate_accuracy, calculate_loss, calculate_precision, calculate_recall

def train_model(model, train_loader, test_loader, num_epochs, learning_rate, device):
    model.to(device)  # Move model to the specified device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print("Epoch Number: " + str(epoch+1))
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()  # Zero the gradients
            output = model(data)  # Forward pass

            loss = calculate_loss(output, target, criterion)  # Calculate loss
            loss.backward()  # Backward pass and optimize
            optimizer.step()

            running_loss += loss.item()

        # Calculate accuracy, precision, and recall on test data
        test_accuracy = calculate_accuracy(model, test_loader, device)

        # Get predictions and targets for precision and recall calculations
        all_predicted = []
        all_targets = []
        model.eval()  # Set model to evaluation mode for predictions

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                all_predicted.append(predicted)
                all_targets.append(target)

        all_predicted = torch.cat(all_predicted)
        all_targets = torch.cat(all_targets)

        # Calculate precision and recall
        test_precision = calculate_precision(all_predicted, all_targets)
        test_recall = calculate_recall(all_predicted, all_targets)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, '
              f'Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, '
              f'Test Recall: {test_recall:.4f}')