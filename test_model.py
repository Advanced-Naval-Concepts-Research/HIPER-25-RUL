import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import matplotlib.pyplot as plt


def test_model(model:nn.Module, criterion, dataset:DataLoader, device="cpu"):
    # Testing loop

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataset, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            predicted = torch.round(outputs).int()
            predicted = torch.reshape(predicted, (-1,))

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(dataset)

    # print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def test_encoder(model: nn.Module, test_loader: DataLoader):
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            
            # Move data to device and reshape like in training
            data = batch[0].to(device)
            data = data.reshape(data.shape[0] * data.shape[1], -1)
            
            # Forward pass
            reconstructed = model(data)
            
            # Calculate loss
            loss = criterion(reconstructed, data)
            
            # Accumulate loss and sample count
            total_loss += loss.item() * data.size(0)  # Multiply by number of samples in batch
            total_samples += data.size(0)

    avg_loss = total_loss / total_samples
    print(f"\nTest Loss: {avg_loss:.4f}")
    return avg_loss