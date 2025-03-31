# Trains a specified model on specified dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import pickle
import matplotlib.pyplot as plt

import sys

'''
Params dictionary:
{
    "criterion": pytorch loss function,
    "optimizer": pytorch optimizer,
    "scheduler": pytorch learning rate scheduler,
    "num_epochs": int
}
'''


def train_model(model:nn.Module, criterion, optimizer:optim.Optimizer, dataset:DataLoader, val_dataset:DataLoader, num_epochs=25, scheduler:optim.lr_scheduler=None, device="cpu"):

    # Training loop
    model_weights = []
    val_losses = []
    train_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for sequences, targets in dataset:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            outputs = outputs.reshape([-1])
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        
        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}, Loss: {running_loss}")
        train_losses.append(running_loss)

        if scheduler is not None:
            scheduler.step()

        model_weights.append(model.state_dict())

        # Validation
        val_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for sequences, targets in val_dataset:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                outputs = outputs.reshape([-1])
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                predicted = torch.round(outputs).int()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss}, Accuracy: {correct/total}")

    # Return model weights that minimize validation loss
    min_val_loss = min(val_losses)
    min_val_loss_idx = val_losses.index(min_val_loss)
    model.load_state_dict(model_weights[min_val_loss_idx])

    # print(min_val_loss, min_val_loss_idx)
    # plt.plot(val_losses, label="val")
    # plt.plot(train_losses, label="train")
    # plt.legend()
    # plt.show()
    return model.state_dict()



# train_model_start function
# Arguments:
#   model: The model to train
#   params: A dictionary containing the parameters for training
# Returns: Trained model weights
# def train_model_start(model:nn.Module, params:dict, dataset:DataLoader):

#     # Unpack parameters
#     criterion = params["criterion"]
#     optimizer = params["optimizer"]
#     scheduler = params["scheduler"]
#     num_epochs = params["num_epochs"]

#     # Set model to training mode
#     model.train(True)
    
#     weights = train_model(model, criterion, optimizer, dataset, num_epochs, scheduler)

#     return weights

    # WILL ACTUALLY SAVE WEIGHTS IN OTHER FILES
    # Save weights to file
    # pickle.dump(weights, open("models/model_weights/" + model.get_name() + "/" + model.get_name() + "_weights.pkl", "wb"))


def train_encoder(model:nn.Module, dataset:DataLoader, num_epochs=5, learning_rate:float = 1e-3):
     criterion = nn.MSELoss()  # Paper didnt specify, will use most common
     optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Paper didnt specify, will use most common
     # Training loop
     
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     model.to(device)
     
     for epoch in range(num_epochs):
         total_loss = 0.0
         
         for batch in dataset:
             data_batch = batch[0].reshape(batch[0].shape[0]* batch[0].shape[1], -1).to(device)  # batch is (data, )
             #print(data_batch.shape)
             #assert False #TODO reshape to train encoder on right dimension
             # Forward pass: encode -> decode
             reconstructed = model(data_batch)
             
             # Compute reconstruction loss
             loss = criterion(reconstructed, data_batch)
             
             # Backprop
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()
             
             total_loss += loss.item()
         
         avg_loss = total_loss / len(dataset)
         #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
     return model.state_dict()
