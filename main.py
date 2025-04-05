# Main script function

import pickle as pkl
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data import random_split, Subset

from train_model import train_model as train_model_func
from train_model import train_encoder as train_encoder_func
from test_model import test_model as test_model_func
from test_model import test_encoder as test_encoder_func
from models.overcomplete_autoencoder import OvercompleteAutoencoder as Auto
from data_processing.data_preprocessing import RULDataset as RULDataset
from data_processing.data_preprocessing import create_sensor_groups

from models.base_model import BaseRULModel as Base
from models.CnnLstmDAG import LSTMCNNModel1 as LSTMCNN
from models.CnnLstmDnn import LSTMCNNModel as LSTMCNNAuto
from models.LSTM import LSTMModel as LSTM
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
        #loss=torch.mean(torch.sqrt(torch.sum(torch.square(ground_truth-prediction),axis=-1))) + self.eps
        #return loss
        # replacing with more numerically stable
        se = torch.square(ground_truth - prediction)  # squared error
        sum_se = torch.sum(se, dim=-1)  # sum over feature dim
        root = torch.sqrt(sum_se + self.eps)  # add eps *before* sqrt
        return torch.mean(root)
    
## Custom loss function
class ConservativeLoss(nn.Module):
    def __init__(self, alpha):
        super(ConservativeLoss, self).__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2) + self.alpha*(torch.mean(torch.relu(predictions-targets)))
    
#def init_weights_CNNAUTO(m):
#    if isinstance(m, nn.LSTM):
#        for name, param in m.named_parameters():
#            if 'weight' in name:
#                nn.init.xavier_uniform_(param)  # Xavier uniform initialization
#            elif 'bias' in name:
#                nn.init.zeros_(param)  # Zero initialization for biases

def load_dataset(sequence_size:int, op_prof:int, sensor_group:str, interpolation:int=None, min_max=False, autodim:int=None, partition:str = "A"):
    # Loads dataset based on parameters and returns it

    dir = 'data/processed_data'

    if min_max:
        dir += '/min_max'

    if autodim is not None:
        assert partition == "A" or partition =="B" or partition =="C"
        dir += "/Autoencoder/14->" +str(autodim) + "/" + partition
    else:
        if interpolation is not None:
            dir += '/interpolated/' + str(interpolation)
        else:
            dir += '/original'

    file = dir + "/" + sensor_group + "/op_prof_" + str(op_prof) + '/dataset_seq_' + str(sequence_size) + '.pt'

    dataset = torch.load(file, weights_only=False)


    class ZScoreTransform():
        def __call__(self, sample):
            # mean = sample.mean(axis=0)
            # std = sample.std(axis=0)
            # return (sample - mean) / std
            return sample - sample[0,:]

    # if interpolation == 30:
    #     dataset = RULDataset(dataset.sensor_data, dataset.rul_labels, 10, transform=ZScoreTransform())
    # else:
    # dataset = RULDataset(dataset.sensor_data, dataset.rul_labels, None, transform=ZScoreTransform())


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

def train_model(model_name:str, hyperparameters:dict, sensor_groups:dict, partition:str="A", min_max:bool=False):
    # Functiont to control training
    # Takes in a model, and hyperparameters
    # Creates necessary objects and trains model
    # For each sequence
    #    For each sensor group
    # Saves model weights to folder

    fp_to_idx = map_fp_to_idx()

    for data_idx in tqdm(range(1,51)):


        with open("data/train_test_val_sets/partition_" + partition + "/split_" + str(data_idx) + ".pkl", "rb") as f:
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
                    if model_name == "Base" or model_name == "LSTM":
                        dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=None, min_max=min_max)
                    elif model_name == "LSTMCNN":
                        dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=30, min_max=min_max)
                    elif model_name == "LSTMCNNAuto":
                        # currently this is right dataset
                        dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=14, min_max=min_max, autodim = 50, partition = partition)
                    elif model_name == "Auto":
                        dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=14, min_max=min_max)

                    model = None
                    if model_name == "Base":
                        model = Base(len(sensors))
                    elif model_name == "LSTMCNN":
                        model = LSTMCNN(len(sensors))
                    elif model_name == "LSTMCNNAuto":
                        model = LSTMCNNAuto(len(sensors))
                        #init_weights_CNNAUTO(model.lstm1)
                        #init_weights_CNNAUTO(model.lstm2)
                    elif model_name == "LSTM":
                        # model = LSTM(len(sensors))
                        model = LSTM(len(sensors))
                    elif model_name =="Auto":
                        # from len(sensors) features to 50-dim features
                        model = Auto(len(sensors), 50)
                    else:
                        raise ValueError(f"Invalid model name:{model_name}")
                    
                    # Move model to device
                    model.to(device)
                    
                    # Dataloader
                    # Split into train and validation
                    train_dataset = Subset(dataset, train_fps)
                    val_dataset = Subset(dataset, val_fps)
                    train_loader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)

                    # Create optimizer
                    optimizer = hyperparameters["optimizer"](model.parameters(), lr=hyperparameters["learning_rate"], 
                                                             weight_decay = hyperparameters["weight_decay"])
                    loss_fn = hyperparameters["loss_fn"]
                    if hyperparameters["scheduler"] is not None:
                        scheduler = hyperparameters["scheduler"](optimizer)
                    else:
                        scheduler = None

                    # Train model
                    if model.get_name() == "Auto":
                        assert False, "should not be here"
                    else:
                        model_weights = train_model_func(model, criterion=loss_fn, optimizer=optimizer, dataset=train_loader, val_dataset=val_loader, num_epochs=hyperparameters["num_epochs"], scheduler=scheduler, device=device)
                    
                    # Save model weights
                    dir = "models/model_weights/" + model_name + "/"
                    if min_max:
                        dir += "min_max/"
                    dir += partition + "/" + sensor_group + "/" + str(sequence_size) + "/"
                    os.makedirs(dir, exist_ok=True)
                    torch.save(model_weights, dir + "set_" + str(data_idx) + "_op_prof_" + str(op_prof) + "_seq_" + str(sequence_size) + ".pt")


def test_model(model_name, hyperparameters:dict, sensor_groups:list, partition:str, min_max:bool=False):
    # Function to control testing
    fp_to_idx = map_fp_to_idx()

    test_loss = {sensor_name:{i:{j:[] for j in [4,5,6]} for i in [1,2,3]} for sensor_name in sensor_groups.keys()}

    test_accuracy = {sensor_name:{i:{j:[] for j in [4,5,6]} for i in [1,2,3]} for sensor_name in sensor_groups.keys()}

    for data_idx in tqdm(range(1,51)):

        with open("data/train_test_val_sets/partition_" + partition + "/split_" + str(data_idx) + ".pkl", "rb") as f:
            train_test_val_set = pkl.load(f)
            test_fps = train_test_val_set["test"]
            test_fps = [fp_to_idx[i] for i in test_fps]

        for sequence_size in [6,5,4]:
            for op_prof in [1, 2, 3]:
                for sensor_group, sensors in sensor_groups.items():

                    # Load dataset
                    if model_name == "Base" or model_name == "LSTM":
                        dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=None, min_max=min_max)
                    elif model_name == "LSTMCNN":
                        dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=30, min_max=min_max)
                    elif model_name == "LSTMCNNAuto":
                        dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=14, min_max=min_max)
                    

                    model = None
                    if model_name == "Base":
                        model = Base(len(sensors))
                    elif model_name == "LSTMCNN":
                        model = LSTMCNN(len(sensors))
                    elif model_name == "LSTMCNNAuto":
                        model = LSTMCNNAuto(len(sensors))
                    elif model_name == "LSTM":
                        model = LSTM(len(sensors), sequence_size)
                    else:
                        raise ValueError("Invalid model name")
                    
                    # Load model dict
                    filepath = "models/model_weights/" + model_name + "/"
                    if min_max:
                        filepath += "min_max/"
                    filepath += partition + "/" + sensor_group + "/" + str(sequence_size) + "/"
                    filepath += "set_" + str(data_idx) + "_op_prof_" + str(op_prof) + "_seq_" + str(sequence_size) + ".pt"
                    # filepath = "models/model_weights/" + model_name + "/" + partition + "/" + sensor_group + "/" + str(sequence_size) + "/set_" + str(data_idx) + "_op_prof_" + str(op_prof) + "_seq_" + str(sequence_size) + ".pt"
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



def train_encoder(hyperparameters:dict, sensor_groups:dict, partition:str="A", min_max:bool=False):
    fp_to_idx = map_fp_to_idx()
    
    with open("data/train_test_val_sets/partition_" + partition + "/split_" + str(1) + ".pkl", "rb") as f:
            train_test_val_set = pkl.load(f)
            #print(train_test_val_set)
            #assert False
            train_fps = train_test_val_set["train"]
            val_fps = train_test_val_set["val"]

            # Sub to indices
            train_fps = [fp_to_idx[i] for i in train_fps]
            val_fps = [fp_to_idx[i] for i in val_fps]

            for sequence_size in [6,5,4]:
                for op_prof in [1, 2, 3]:
                    for sensor_group, sensors in sensor_groups.items():
                        dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=14, min_max=min_max)
                        model = Auto(len(sensors), 50)
                        
                        # Move model to device
                        model.to(device)
                        
                        # Dataloader
                        # Split into train and validation
                        train_dataset = Subset(dataset, train_fps)
                        val_dataset = Subset(dataset, val_fps)
                        concat = ConcatDataset([train_dataset,val_dataset])
                        # train the autoencoder on everything but the test data all at once
                        #print(hyperparameters)
                        loader = DataLoader(concat, batch_size=hyperparameters["batch_size"], shuffle=True)

                        # Create optimizer
                        # optimizer = hyperparameters["optimizer"](model.parameters(), lr=hyperparameters["learning_rate"])
                        # loss_fn = hyperparameters["loss_fn"]
                        # if hyperparameters["scheduler"] is not None:
                        #     scheduler = hyperparameters["scheduler"](optimizer)
                        # else:
                        #     scheduler = None

                        # Train model
                        print(f"BEGIN operational profile:{op_prof}, sequencesize: {sequence_size}")
                        model_weights = train_encoder_func(model, loader)
                        
                        # Save model weights
                        dir = "models/model_weights/" + "Auto" + "/"
                        if min_max:
                            dir += "min_max/"
                        dir += partition + "/" + sensor_group + "/" + str(sequence_size) + "/"
                        os.makedirs(dir, exist_ok=True)
                        torch.save(model_weights, dir + "set_" + str(1) + "_op_prof_" + str(op_prof) + "_seq_" + str(sequence_size) + ".pt")

def test_encoder(sensor_groups:dict, partition:str="A"):
    fp_to_idx = map_fp_to_idx()
    test_loss = {sensor_name:{i:{j:[] for j in [4,5,6]} for i in [1,2,3]} for sensor_name in sensor_groups.keys()}
    
    
    with open("data/train_test_val_sets/partition_" + partition + "/split_" + str(1) + ".pkl", "rb") as f:
            train_test_val_set = pkl.load(f)
            test_fps = train_test_val_set["test"]
            test_fps = [fp_to_idx[i] for i in test_fps]

            for sequence_size in [6,5,4]:
                for op_prof in [1, 2, 3]:
                    for sensor_group, sensors in sensor_groups.items():

                        # dataset stuff
                        dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=14, min_max=False)
                        model = Auto(len(sensors), 50)
                        # Load model dict
                        filepath = "models/model_weights/" + "Auto" + "/" + partition + "/" + sensor_group + "/" + str(sequence_size) + "/set_" + str(1) + "_op_prof_" + str(op_prof) + "_seq_" + str(sequence_size) + ".pt"
                        # print(filepath)
                        checkpoint = torch.load(filepath)
                        # print(type(checkpoint))  # This should show 'dict' if it's a valid state_dict
                        # print(checkpoint)  # Inspect the contents of the checkpoint (if it's a dict)
                        model.load_state_dict(checkpoint)

                        # Move model to device
                        model.to(device)

                        # Dataloader
                        test_dataset = Subset(dataset, test_fps)
                        test_loader = DataLoader(test_dataset, batch_size=hyperparameters["batch_size"], shuffle=True)

                        loss = test_encoder_func(model, test_loader)

                        test_loss[sensor_group][op_prof][sequence_size].append(loss)
            return test_loss



def save_statistics(save_dir:str, model:str, minmax:bool, sensor_groups, partition:str, test_loss, test_acc):

    for sensor_group in sensor_groups.keys():
        for op_prof in [1, 2, 3]:

            # Create tables based on op profiles
            # columns = Test #, 4, 5, 6

            if minmax:
                dir = save_dir + "/" + model + "/" + partition + "/minmax/" + sensor_group + "/" + str(op_prof) + "/"
            else:
                    dir = save_dir + "/" + model + "/" + partition + "/regular/" + sensor_group + "/" + str(op_prof) + "/"
            os.makedirs(dir, exist_ok=True)

            df_loss = pd.DataFrame(columns=["Test #", "4", "5", "6"])
            df_loss["Test #"] = [i for i in range(1, 51)]
            df_loss["4"] = test_loss[sensor_group][op_prof][4]
            df_loss["5"] = test_loss[sensor_group][op_prof][5]
            df_loss["6"] = test_loss[sensor_group][op_prof][6]

            df_acc = pd.DataFrame(columns=["Test #", "4", "5", "6"])
            df_acc["Test #"] = [i for i in range(1, 51)]
            df_acc["4"] = test_acc[sensor_group][op_prof][4]
            df_acc["5"] = test_acc[sensor_group][op_prof][5]
            df_acc["6"] = test_acc[sensor_group][op_prof][6]

            # Save to csv
            df_loss.to_csv(dir + "loss.csv", index=False)
            df_acc.to_csv(dir + "accuracy.csv", index=False)



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
        "loss_fn": RMSELoss(),
        # "loss_fn": nn.MSELoss(),
        "scheduler": None,
        "weight_decay" : 0
    }

    return hyperparameters


def setup_hyperparameters_auto():
    
    hyperparameters = {
        "learning_rate": 0.005,
        "num_epochs": 500,
        "batch_size": 20,
        "scheduler": None,
        "weight_decay" : 0
    }

    return hyperparameters

def setup_hyperparameters_lstmcnnAuto():
    # Sets up hyperparameters for training
    # Returns a dictionary of hyperparameters

    hyperparameters = {
        "learning_rate": 0.005,
        "num_epochs": 500,
        "batch_size": 54,
        "optimizer": optim.Adam,
        "loss_fn": RMSELoss(),
        "scheduler": None,
        "weight_decay" : 0.0001
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
        "loss_fn": RMSELoss(),
        "scheduler": None,
        "weight_decay" : 0
    }

    return hyperparameters

def setup_hyperparameters_lstm():

    hyperparameters = {
        "learning_rate": 0.005,
        "num_epochs": 750,
        "batch_size": 54,
        "optimizer": optim.Adam,
        "loss_fn": RMSELoss(),
        "scheduler": None,
        "weight_decay" : 0
    }

    return hyperparameters


def setup_hyperparameters(model:str):
   
    if model == "Base":
        return setup_hyperparameters_base()
    if model == "LSTMCNN":
        return setup_hyperparameters_lstmcnn()
    if model == "LSTM":
        return setup_hyperparameters_lstm()
    if model =="Auto":
        return setup_hyperparameters_auto()
    if model =="LSTMCNNAuto":
        return  setup_hyperparameters_lstmcnnAuto()
    else:
        return "INVALID NAME"

if __name__ == "__main__":
    # Main script function
    # For each model
    #     For each sequence size
    #         For each sensor group
    #             Train model
    #             Test model
    #             Save statistics
    parser = argparse.ArgumentParser(description="Train and Test RUL Models")
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Mode: train or test. Right now the script just trains and this is ignored")
    parser.add_argument("--model", type=str, required=True, choices=["Base", "LSTMCNN", "LSTMCNNAuto", "LSTM", "Auto"], help="Model name")
    parser.add_argument("--partition", type=str, required=True, choices=["A", "B", "C"], help="Training partition")
    parser.add_argument("--minmax", type=str, required=True, choices = ["True", "False"], help="Use minmax constrained data")
    args = parser.parse_args()
    print("Using device:", device)
    
    # Models to train
    # models = ["Base", "LSTMCNN", "LSTMCNNAuto"]
    # models = ["Base"]
    # models = ["LSTMCNN"]
    models = []
    models.append(str(args.model))

    min_max = True if args.minmax == "True" else False 

    print(min_max)
   
    # Sensor groups
    sensor_groups = create_sensor_groups()
    sensor_groups = {"s1_g1": sensor_groups["s1_g1"], "s1_g2": sensor_groups["s1_g2"], "s1_g3": sensor_groups["s1_g3"]}

    for model in models:
        hyperparameters = setup_hyperparameters(model)
        print(hyperparameters)
        if model == "Auto":
            train_encoder(hyperparameters, sensor_groups, str(args.partition))
            end_dict = test_encoder(sensor_groups, str(args.partition))
            print(end_dict)
        else:
            print(args.mode)
            if str(args.mode) == "train":
                train_model(model, hyperparameters, sensor_groups, str(args.partition), min_max)
            else:
                test_model(model, hyperparameters, sensor_groups, str(args.partition), min_max)
                save_statistics("data/stats", model, min_max, sensor_groups, str(args.partition), test_loss, test_acc)
    

    # TODO add testing and statistics production
    # TODO add hyperparameter optimization
    # TODO add saving of statistics