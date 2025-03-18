# Prep datasets for training and testing
# Create train, test, validation splits of the data for specified groupings of sensors

import pandas as pd
import numpy as np
import scipy as sp
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class RULDataset(torch.utils.data.Dataset):
    def __init__(self, sensor_data, rul_labels, sequence_length, transform=None):
        """
        Args:
            sensor_data (array-like): Time-series sensor readings (num_samples, time_steps, sensors).
            rul_labels (array-like): Remaining Useful Life labels (num_samples,).
            sequence_length (int): Length of the sequence to consider for each sample.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.sensor_data = sensor_data
        self.rul_labels = rul_labels
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        # Total number of complete sequences in the dataset
        return len(self.sensor_data) - self.sequence_length + 1

    def __getitem__(self, idx):
        # Retrieve the sequence and the corresponding RUL label
        sequence = self.sensor_data[idx:idx + self.sequence_length, :, :]
        label = self.rul_labels[idx + self.sequence_length - 1]

        # Transformation
        if self.transform:
            sequence = self.transform(sequence)

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# Group 1 (All sensors in cooling and fueling systems): temp delta, cooling flow rate, cooling pump current and pressure delta, fuel lp flow, fuel hp rail, fuel hp relief, fuel lp pressure, fuel hp pressure, fuel lp current, fuel hp current
# Group 2 (Current and pressure sensors only): cooling pump current, cooling pressure delta, fuel lp pressure, fuel hp pressure, fuel lp current, fuel hp current
# Group 3 (current sensors only): cooling pump current, fuel lp current, fuel hp current


# Reads in the sensor values for each opeartional profile from the specified directory and their associated sequence failures
# Returns dict of {operational profile: list of failure profiles and sequence point failures}
def parse_avg_sensor_data(dir:str, seq_failures_filename:str) -> dict:

    data_dict = {1:[], 2:[], 3:[]} # dict of operational profile to list of failure profiles and sequence point failures

    seq_failures = pd.read_csv(seq_failures_filename)["Combined"].tolist() # TODO Once I get the data, may need to parse out based on operational profile 

    # Walk through the directory
    for i in range(1, 101):
        fp_dir = dir + "/Failure_Profile_" + str(i)

        # Read in failure profiles for each operational profile
        for j in range(1,4):
            filename = "AverageValueData_Failure_Profile_" + str(i) + "_Operational_Profile_" + str(j) + ".csv"
            fp_df = pd.read_csv(fp_dir + "/" + filename)
            data_dict[j].append(tuple(fp_df, 0)) # TODO update with correct sequence failure later

    return data_dict




# Create train, test, validation splits of the data for specified groupings of sensors for all data sources over all operational profiles
# Returns dict of {operational profile: PyTorch Dataset of train, test, and validation splits}
def create_train_test_val_splits(data_dict:dict, sensor_group:dict, sequence_length:int, split:list=[0.8, 0.1, 0.1], num_iters:int=1, batch_size:int=4) -> dict:
    """
    data_dict: dict of {operational profile: list of failure profiles and sequence point failures}
    sensor_group: list of sensors to include in the dataset
    sequence_length: length of the sequence to consider for each sample
    split: list of floats representing the split of the data for train, test, and validation
    num_iters: number of times to split the data into train, test, and validation
    """
    
    train_size = split[0] * len(data_dict[1])
    test_size = split[1] * len(data_dict[1])
    val_size = len(data_dict[1]) - train_size - test_size

    # Create Datasets for each operational profile
    datasets = {1: [], 2: [], 3: []}
    for i in range(1, 4):
        op_prof_data = []
        op_prof_labels = []
        for j in range(len(data_dict[i])):
            sensor_data = data_dict[i][j][0][sensor_group].to_numpy()
            rul_labels = data_dict[i][j][1].to_numpy()
            op_prof_data.append(sensor_data)
            op_prof_labels.append(rul_labels)

        dataset = RULDataset(np.array(op_prof_data), np.array(op_prof_labels), sequence_length)
        datasets[i] = dataset

    # Split the data into train, test, and validation for each operational profile for number_split times
    dataloaders = {1: [], 2: [], 3: []}
    for i in range(num_iters):
        for j in range(1, 4):
            train, test, val = random_split(datasets[j], [train_size, test_size, val_size])
            train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
            dataloaders[j].append((train_loader, test_loader, val_loader))

    return datasets, dataloaders

# Creates sensor groups based on the specified sensors TODO
# Since plants are independent of each other, we can create separate sensor groups for each plant
def create_sensor_groups():
    sensors = {}

    # Group 1 TODO
    sensors["s1_g1"] = []
    sensors["s2_g1"] = []

    # Group 2
    sensors["s1_g2"] = []
    sensors["s2_g2"] = []

    # Group 3
    sensors["s1_g3"] = []
    sensors["s2_g3"] = []

    return sensors

# Loads data and places them into PyTorch Dataset and Dataloader objects for all sensor groups
def get_data(data_dir, sequences_filename, num_iters=1, batch_size=4):
    data_dict = parse_avg_sensor_data(data_dir, sequences_filename)

    sensor_groups = create_sensor_groups()

    sensor_to_datasets = {}

    for group_id, sensor_group in sensor_groups.items():
        datasets, dataloaders = create_train_test_val_splits(data_dict, sensor_group, 30, [0.8, 0.1, 0.1], num_iters=num_iters, batch_size=batch_size)
        sensor_to_datasets[group_id] = (datasets, dataloaders)

    return sensor_to_datasets

# Saves datasets and dataloaders to files
def save_data(sensor_to_datasets, save_dir):
    
    for sensor_group, t in sensor_to_datasets.items():
        datasets, dataloaders = t

        # Create dir
        dir = save_dir + "/" + sensor_group
        os.makedirs(dir, exist_ok=True)

        # Save datasets to dir
        for dataset in datasets:
            torch.save(dataset, dir + "/" + "dataset_" + str(dataset) + ".pt")

# Performs a 2nd degree polynomial interpolation on the specified dataset
# Returns the interpolated dataset
def polynomial_interpolation(dataset, num_points):
    pass


if __name__ == "__main__":
    data_dict = parse_avg_sensor_data("data/AvgValue_Data", "data/Failure_Profile_Labels/labels.csv")
    sensor_group = create_sensor_groups()

    datasets, dataloaders = create_train_test_val_splits(data_dict, sensor_group, 4, [0.8, 0.1, 0.1], num_iters=1, batch_size=8)
    pass