# Trains a specified model on specified dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import pickle

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

#
def train_encoder(model:nn.Module, dataset:DataLoader, num_epochs=5, learning_rate:float = 1e-3):
    criterion = nn.MSELoss()  # Paper didnt specify, will use most common
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Paper didnt specify, will use most common
    # Training loop
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch in dataset:
            data_batch = batch[0]  # batch is (data, )
            
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

def train_model(model:nn.Module, criterion, optimizer:optim.Optimizer, dataset:DataLoader, num_epochs=25, scheduler:optim.lr_scheduler=None):
    
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0

        for sequences, targets in dataset:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}, Loss: {running_loss}")

        if scheduler is not None:
            scheduler.step()

    return model.state_dict()



# train_model_start function
# Arguments:
#   model: The model to train
#   params: A dictionary containing the parameters for training
# Returns: Trained model weights
def train_model_start(model:nn.Module, params:dict, dataset:DataLoader):

    # Unpack parameters
    criterion = params["criterion"]
    optimizer = params["optimizer"]
    scheduler = params["scheduler"]
    num_epochs = params["num_epochs"]

    # Set model to training mode
    model.train(True)
    
    weights = train_model(model, criterion, optimizer, dataset, num_epochs, scheduler)

    return weights

    # WILL ACTUALLY SAVE WEIGHTS IN OTHER FILES
    # Save weights to file
    # pickle.dump(weights, open("models/model_weights/" + model.get_name() + "/" + model.get_name() + "_weights.pkl", "wb"))

