# Main script function

import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split

from train_model import train_model as tm

from data_processing.data_preprocessing import RULDataset as RULDataset
from data_processing.data_preprocessing import create_sensor_groups

from models.base_model import BaseRULModel as Base
from models.CnnLstmDAG import LSTMCNNModel1 as LSTMCNN
from models.CnnLstmDnn import LSTMCNNModel as LSTMCNNAuto
# TODO add other LSTM model

# Hyperparameter optimization
# Model training
# Model testing
# Production of statistics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(sequence_size:int, op_prof:int, sensor_group:str, interpolation:int=None):
    # Loads dataset based on parameters and returns it

    dir = 'data/processed_data'

    if interpolation is not None:
        dir += '/interpolated/' + str(interpolation)
    else:
        dir += '/original'

    file = dir + "/" + sensor_group + "/op_prof_" + str(op_prof) + '/dataset_seq_' + str(sequence_size) + '.pt'

    dataset = torch.load(file, weights_only=False)

    return dataset

def train_model(model_name:str, hyperparameters:dict, sensor_groups:list):
    # Functiont o control training
    # Takes in a model, and hyperparameters
    # Creates necessary objects and trains model
    # For each sequence
    #    For each sensor group
    # Saves model weights to folder

    # TODO Need to load in the set train, test, validation split data sets
    
    # TODO Did Andy keep any completely separate for testing?
    # From the thesis, it seems like he didn't. Need to check with him

    test_loss = {}
    test_accuracy = {}
    for sequence_size in [6,5,4]:
        for op_prof in [1, 2, 3]:
            for sensor_group, sensors in sensor_groups.items():
                dataset = load_dataset(sequence_size, op_prof, sensor_group)
                model = None
                if model_name == "Base":
                    model = Base(len(sensors))
                elif model_name == "LSTMCNN":
                    model = LSTMCNN(len(sensors))
                elif model_name == "LSTMCNNAuto":
                    model = LSTMCNNAuto(len(sensors))
                elif model_name == "LSTM":
                    # model = LSTM(len(sensors))
                    pass
                    # TODO add LSTM model
                else:
                    raise ValueError("Invalid model name")
                
                # Move model to device
                model.to(device)
                
                # Dataloader
                # Split into train, test, validation
                train_size = int(0.8 * len(dataset))
                val_size = int(0.1 * len(dataset))
                test_size = len(dataset) - train_size - val_size
                train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

                train_loader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)

                # train_loader = DataLoader(dataset, batch_size=hyperparameters["batch_size"], shuffle=True)
                # val_loader = DataLoader(dataset, batch_size=hyperparameters["batch_size"], shuffle=False)

                # Create optimizer
                # optimizer = hyperparameters["optimizer"](model.parameters(), lr=hyperparameters["learning_rate"])
                # loss_fn = hyperparameters["loss_fn"]
                optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
                loss_fn = nn.MSELoss()
                if hyperparameters["scheduler"] is not None:
                    scheduler = hyperparameters["scheduler"](optimizer)
                else:
                    scheduler = None

                # Train model
                model_weights = tm(model, criterion=loss_fn, optimizer=optimizer, dataset=train_loader, val_dataset=val_loader, num_epochs=hyperparameters["num_epochs"], scheduler=scheduler, device=device)
                
                # Save model weights
                dir = "models/model_weights/" + model_name + "/" + sensor_group + "/"
                os.makedirs(dir, exist_ok=True)
                torch.save(model_weights, dir + "op_prof_" + str(op_prof) + "_seq_" + str(sequence_size) + ".pt")

                # Test model
                # TODO
                # test_loss, test_accuracy = test_model(model, test_loader, loss_fn)
                test_loss = 0.0
                correct = 0
                total = 0

                model.load_state_dict(model_weights)
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        outputs = outputs.reshape([-1])
                        loss = loss_fn(outputs, labels)
                        test_loss += loss.item()

                        # TODO round?
                        # _, predicted = torch.max(outputs, 1)
                        # total += labels.size(0)
                        # correct += (predicted == labels).sum().item()
                        print(outputs, labels)

                print(correct, total)



# For each mode
#     For each sequence
#         For each sensor group


# Hyperparameter optimization
# Hyperparameters contain the following:
#     Learning rate
#     Number of epochs
#     Batch size
#     Optimizer
#     Loss function
#     Scheduler (Optional)
def setup_hyperparameters_base():
    # Sets up hyperparameters for training
    # Returns a dictionary of hyperparameters

    hyperparameters = {
        "learning_rate": 0.005,
        "num_epochs": 500,
        "batch_size": 56,
        "optimizer": optim.Adam,
        "loss_fn": nn.MSELoss(),
        "scheduler": None
    }

    return hyperparameters


if __name__ == "__main__":
    # Main script function
    # For each model
    #     For each sequence size
    #         For each sensor group
    #             Train model
    #             Test model
    #             Save statistics

    print("Using device:", device)

    # Models to train
    # models = ["Base", "LSTMCNN", "LSTMCNNAuto"]
    models = ["Base"]

    # Sensor groups
    sensor_groups = create_sensor_groups()
    sensor_groups = {"s1_g1": sensor_groups["s1_g1"]}

    # Hyperparameters
    hyperparameters = setup_hyperparameters_base()

    for model in models:
        train_model(model, hyperparameters, sensor_groups)
    
    # TODO add testing and statistics production
    # TODO add hyperparameter optimization
    # TODO add saving of statistics