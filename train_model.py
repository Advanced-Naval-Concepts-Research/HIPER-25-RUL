# Trains a specified model on the dataset

import torch
import torch.nn as nn
import torch.optim as optim

'''
Params dictionary:
{
    "criterion": pytorch loss function,
    "optimizer": pytorch optimizer,
    "scheduler": pytorch learning rate scheduler,
    "num_epochs": int,
    "lr": float,
    "batch_size": int,
    "device": torch.device,
}
'''


def train_model(model:nn.Module, criterion, optimizer:optim.Optimizer, scheduler:optim.lr_scheduler, num_epochs=25):
    
    return # Model Weights



# Main function
# Arguments:
#   model: The model to train
#   params: A dictionary containing the parameters for training
# Returns: Trained model weights
def main(model:nn.Module, params:dict):

    # Unpack parameters
    criterion = params["criterion"]
    optimizer = params["optimizer"]
    scheduler = params["scheduler"]
    num_epochs = params["num_epochs"]
    lr = params["lr"]
    batch_size = params["batch_size"]
    device = params["device"]

    # Set model to training mode
    model.train(True)
    
    weights = train_model(model, criterion, optimizer, scheduler, num_epochs)

    pass