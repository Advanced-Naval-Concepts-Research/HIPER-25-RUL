# Main script function

import pickle as pkl
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split, Subset

from train_model import train_model as train_model_func
from test_model import test_model as test_model_func

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

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps=1e-6

    def forward(self,ground_truth,prediction):
        loss=torch.mean(torch.sqrt(torch.sum(torch.square(ground_truth-prediction),axis=-1))) + self.eps
        return loss
    


def load_dataset(sequence_size:int, op_prof:int, sensor_group:str, interpolation:int=None):
    # Loads dataset based on parameters and returns it

    dir = 'data/processed_data'

    if interpolation is not None:
        dir += '/interpolated/' + str(interpolation)
    else:
        dir += '/original'

    file = dir + "/" + sensor_group + "/op_prof_" + str(op_prof) + '/dataset_seq_' + str(sequence_size) + '.pt'

    dataset = torch.load(file, weights_only=False)

    # Convert to zscore
    # mean = dataset.sensor_data.mean(axis=1)
    # std = dataset.sensor_data.std(axis=1)

    # class ZScoreTransform():
    #     def __call__(self, sample):
    #         # mean = sample.mean(axis=0)
    #         # std = sample.std(axis=0)
    #         # return (sample - mean) / std
    #         return sample - sample[0,:]

    # if interpolation == 30:
    #     dataset = RULDataset(dataset.sensor_data, dataset.rul_labels, 3, transform=ZScoreTransform())
    # else:
    #     dataset = RULDataset(dataset.sensor_data, dataset.rul_labels, None, transform=ZScoreTransform())


    return dataset

def map_fp_to_idx():
    # Based on the labels, map the failure profiles to indices
    sequences_csv = pd.read_csv("data/Failure_Profile_Labels/labels_combined.csv")

    num_leak = 0
    fp_to_idx = {}
    for _, row in sequences_csv.iterrows():
        fp = row[0]
        seq_a = row[1]
        seq_b = row[2]
        seq_c = row[3]

        if seq_a == -1 or seq_b == -1 or seq_c == -1:
            num_leak += 1
            continue

        fp_to_idx[fp] = int(fp) - 1 - num_leak

    return fp_to_idx

def train_model(model_name:str, hyperparameters:dict, sensor_groups:dict):
    # Functiont o control training
    # Takes in a model, and hyperparameters
    # Creates necessary objects and trains model
    # For each sequence
    #    For each sensor group
    # Saves model weights to folder

    fp_to_idx = map_fp_to_idx()

    for data_idx in tqdm(range(1,51)):

        with open("data/train_test_val_sets/partition_A/split_" + str(data_idx) + ".pkl", "rb") as f:
            train_test_val_set = pkl.load(f)
            train_fps = train_test_val_set["train"]
            val_fps = train_test_val_set["val"]

            # Sub to indices
            train_fps = [fp_to_idx[i] for i in train_fps]
            val_fps = [fp_to_idx[i] for i in val_fps]

        for sequence_size in [6,5,4]:
            for op_prof in [1, 2, 3]:
                for sensor_group, sensors in sensor_groups.items():

                    # Load dataset
                    if model_name == "Base":
                        dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=None)
                    elif model_name == "LSTMCNN":
                        dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=30)
                    elif model_name == "LSTMCNNAuto":
                        dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=14)
                    elif model_name == "LSTM":
                        # TODO
                        pass

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
                    # Split into train and validation
                    train_dataset = Subset(dataset, train_fps)
                    val_dataset = Subset(dataset, val_fps)
                    train_loader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)

                    # Create optimizer
                    optimizer = hyperparameters["optimizer"](model.parameters(), lr=hyperparameters["learning_rate"])
                    loss_fn = hyperparameters["loss_fn"]
                    if hyperparameters["scheduler"] is not None:
                        scheduler = hyperparameters["scheduler"](optimizer)
                    else:
                        scheduler = None

                    # Train model
                    model_weights = train_model_func(model, criterion=loss_fn, optimizer=optimizer, dataset=train_loader, val_dataset=val_loader, num_epochs=hyperparameters["num_epochs"], scheduler=scheduler, device=device)
                    
                    # Save model weights
                    dir = "models/model_weights/" + model_name + "/" + sensor_group + "/" + str(sequence_size) + "/"
                    os.makedirs(dir, exist_ok=True)
                    torch.save(model_weights, dir + "set_" + str(data_idx) + "_op_prof_" + str(op_prof) + "_seq_" + str(sequence_size) + ".pt")


def test_model(model_name, hyperparameters:dict, sensor_groups:list):
    # Function to control testing
    fp_to_idx = map_fp_to_idx()

    test_loss = {sensor_name:{i:{j:[] for j in [4,5,6]} for i in [1,2,3]} for sensor_name in sensor_groups.keys()}

    test_accuracy = {sensor_name:{i:{j:[] for j in [4,5,6]} for i in [1,2,3]} for sensor_name in sensor_groups.keys()}

    for data_idx in tqdm(range(1,51)):

        with open("data/train_test_val_sets/partition_A/split_" + str(data_idx) + ".pkl", "rb") as f:
            train_test_val_set = pkl.load(f)
            test_fps = train_test_val_set["test"]
            test_fps = [fp_to_idx[i] for i in test_fps]

        for sequence_size in [6,5,4]:
            for op_prof in [1, 2, 3]:
                for sensor_group, sensors in sensor_groups.items():

                    # Load dataset
                    if model_name == "Base":
                        dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=None)
                    elif model_name == "LSTMCNN":
                        dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=30)
                    elif model_name == "LSTMCNNAuto":
                        dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=14)
                    elif model_name == "LSTM":
                        # TODO
                        pass

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
                    
                    # Load model dict
                    filepath = "models/model_weights/" + model_name + "/" + sensor_group + "/" + str(sequence_size) + "/set_" + str(data_idx) + "_op_prof_" + str(op_prof) + "_seq_" + str(sequence_size) + ".pt"
                    model.load_state_dict(torch.load(filepath))

                    # Move model to device
                    model.to(device)

                    # Dataloader
                    test_dataset = Subset(dataset, test_fps)
                    test_loader = DataLoader(test_dataset, batch_size=hyperparameters["batch_size"], shuffle=True)

                    loss_fn = hyperparameters["loss_fn"]
                    loss, acc = test_model_func(model, loss_fn, test_loader, device)

                    test_loss[sensor_group][op_prof][sequence_size].append(loss)
                    test_accuracy[sensor_group][op_prof][sequence_size].append(acc)

    
    # Print out averages
    for sequence_size in [6,5,4]:
            print("\n\n\nSequence:", sequence_size)
            for op_prof in [1, 2, 3]:
                for sensor_group, sensors in sensor_groups.items():
                    print("Op prof:", op_prof, ". Sensor group", sensor_group)
                    avg_acc = torch.mean(torch.tensor(test_accuracy[sensor_group][op_prof][sequence_size]))
                    avg_loss = torch.mean(torch.tensor(test_loss[sensor_group][op_prof][sequence_size]))

                    print("Avg accuracy: ", avg_acc, ". Avg loss: ", avg_loss)
    
    return test_loss, test_accuracy


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
        "batch_size": 54,
        "optimizer": optim.Adam,
        # "loss_fn": RMSELoss(),
        "loss_fn": nn.MSELoss(),
        "scheduler": None
    }

    return hyperparameters


def setup_hyperparameters_lstmcnn():
    # Sets up hyperparameters for training
    # Returns a dictionary of hyperparameters

    hyperparameters = {
        "learning_rate": 0.005,
        "num_epochs": 500,
        "batch_size": 54,
        "optimizer": optim.RMSprop,
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
    # models = ["Base"]
    models = ["LSTMCNN"]

    # Sensor groups
    sensor_groups = create_sensor_groups()
    sensor_groups = {"s1_g1": sensor_groups["s1_g1"], "s1_g2": sensor_groups["s1_g2"], "s1_g3": sensor_groups["s1_g3"]}

    # Hyperparameters
    hyperparameters = setup_hyperparameters_base()

    for model in models:
        train_model(model, hyperparameters, sensor_groups)
        test_model(model, hyperparameters, sensor_groups)
    
    # TODO add testing and statistics production
    # TODO add hyperparameter optimization
    # TODO add saving of statistics