# Prep datasets for training and testing
# Create train, test, validation splits of the data for specified groupings of sensors

import pandas as pd
import numpy as np
import scipy as sp
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class RULDataset(torch.utils.data.Dataset):
    def __init__(self, sensor_data, rul_labels, sequence_length=None, transform=None):
        """
        Args:
            sensor_data (array-like): Time-series sensor readings (num_samples, time_steps, sensors).
            rul_labels (array-like): Remaining Useful Life labels (num_samples,).
            sequence_length (int): Length of the sequence to consider for each sample.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.sensor_data = sensor_data
        self.rul_labels = rul_labels
        self.transform = transform
        if sequence_length:
            self.sequence_length = sequence_length

            # Reshape data to be (num_samples, time_steps, sequence_length, sensors)
            t_new = sensor_data.shape[1] // self.sequence_length
            self.sensor_data = sensor_data.reshape(sensor_data.shape[0], t_new, self.sequence_length, sensor_data.shape[2])

    def __len__(self):
        # Total number of complete sequences in the dataset
        return len(self.sensor_data)

    def __getitem__(self, idx):
        # Retrieve the sequence and the corresponding RUL label
        sequence = self.sensor_data[idx]
        label = self.rul_labels[idx]

        # Apply transformation if specified
        if self.transform:
            sequence = self.transform(sequence)

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Reads in the sensor values for each opeartional profile from the specified directory and their associated sequence failures
# Only parses over labeled data. In this case, labeled data are failures where clogs occur
# Returns dict of {operational profile: list of failure profiles and sequence point failures}
def parse_avg_sensor_data(dir:str, seq_failures_filename:str) -> dict:

    data_dict = {1:[], 2:[], 3:[]} # dict of operational profile to list of failure profiles and sequence point failures

    seq_failures = pd.read_csv(seq_failures_filename)

    # Walk through the directory
    for idx, row in seq_failures.iterrows():
        i = row["Failure Profile"]
        fp_dir = dir + "/Failure_Profile_" + str(i)

        # Read in failure profiles for each operational profile
        for j in range(1,4):
            filename = "AverageValueData_Failure_Profile_" + str(i) + "_Operational_Profile_" + str(j) + ".csv"
            fp_df = pd.read_csv(fp_dir + "/" + filename)
            data_dict[j].append(tuple([fp_df, row.iloc[j]]))

    return data_dict




# Create train, test, validation splits of the data for specified groupings of sensors for all data sources over all operational profiles
# Returns dict of {operational profile: PyTorch Dataset of train, test, and validation splits}
def create_train_test_val_splits(data_dict:dict, sensor_group:dict, split:list=[0.8, 0.1, 0.1], num_iters:int=1, batch_size:int=4) -> dict:
    """
    data_dict: dict of {operational profile: list of failure profiles and sequence point failures}
    sensor_group: list of sensors to include in the dataset
    sequence_length: length of the sequence to consider for each sample
    split: list of floats representing the split of the data for train, test, and validation
    num_iters: number of times to split the data into train, test, and validation
    """
    
    train_size = int(split[0] * len(data_dict[1]))
    test_size = int(split[1] * len(data_dict[1]))
    val_size = len(data_dict[1]) - train_size - test_size

    # Create Datasets for each operational profile
    datasets = {1: [], 2: [], 3: []}
    for i in range(1, 4): # For each operational profile
    
        for sequence_length in range(4,7): # For each sequence length

            op_prof_data = []
            op_prof_labels = []
            for j in range(len(data_dict[i])):
                sensor_data = data_dict[i][j][0][sensor_group].to_numpy()[:sequence_length]
                rul_labels = data_dict[i][j][1]
                op_prof_data.append(sensor_data)
                op_prof_labels.append(rul_labels)

            dataset = RULDataset(np.array(op_prof_data), np.array(op_prof_labels))
            datasets[i].append(dataset)

    # Split the data into train, test, and validation for each operational profile for number_split times
    dataloaders = {1: [], 2: [], 3: []}
    for i in range(num_iters):
        for j in range(1, 4): # Operational profile
            for k in range(3): # Sequence length
                train, test, val = random_split(datasets[j][k], [train_size, test_size, val_size])
                train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
                dataloaders[j].append((train_loader, test_loader, val_loader))

    return datasets, dataloaders

# Creates sensor groups based on the specified sensors
# Since plants are independent of each other, we can create separate sensor groups for each plant
# Group 1 (All sensors in cooling and fueling systems): temp delta, cooling flow rate, cooling pump current and pressure delta, fuel lp flow, fuel hp rail, fuel hp relief, fuel lp pressure, fuel hp pressure, fuel lp current, fuel hp current
# Group 2 (Current and pressure sensors only): cooling pump current, cooling pressure delta, fuel lp pressure, fuel hp pressure, fuel lp current, fuel hp current
# Group 3 (current sensors only): cooling pump current, fuel lp current, fuel hp current
def create_sensor_groups():
    sensors = {}

    # Group 1
    sensors["s1_g1"] = ["Fuel System 1  LP 1 ", "Cooling System 1 P 1", "Cooling System 1 P 2", "Cooling System 1 Heater Input Temperature ", "Cooling System 1 Heater Output Temperature", "Fuel System 1 High Pressure Rail", "Cooling System 1 Flow", "Fuel System 1  Injector Flow", "Fuel System 1  Service Flow", "Fuel System 1  HP Relief Flow", "Fuel System 1  Injector Pump ", "Fuel System 1  Service Pump ", "Cooling System 1 Service Pump "]
    sensors["s2_g1"] = ["Fuel System 2  LP 1", "Cooling System 2 P 1", "Cooling System 2 P 2", "Cooling System 2  Heater Output Temperature", "Cooling System 2 Heater Input Temperature ", "Fuel System 2  High Pressure Rail", "Fuel System 2  Injector Flow", "Cooling System 2 Flow", "Fuel System 2  Service Flow", "Fuel System 2  HP Relief Flow", "Fuel System 2  Injector Pump ", "Fuel System 2  Service Pump ", "Cooling System 2 Service Pump "]
    sensors["combined_g1"] = sensors["s1_g1"] + sensors["s2_g1"]

    # Group 2
    sensors["s1_g2"] = ["Fuel System 1  LP 1 ", "Cooling System 1 P 1", "Cooling System 1 P 2", "Fuel System 1 High Pressure Rail", "Fuel System 1  Injector Pump ", "Fuel System 1  Service Pump ", "Cooling System 1 Service Pump "]
    sensors["s2_g2"] = ["Fuel System 2  LP 1", "Cooling System 2 P 1", "Cooling System 2 P 2", "Fuel System 2  High Pressure Rail","Fuel System 2  Injector Pump ", "Fuel System 2  Service Pump ", "Cooling System 2 Service Pump "]
    sensors["combined_g2"] = sensors["s1_g2"] + sensors["s2_g2"]

    # Group 3
    sensors["s1_g3"] = ["Fuel System 1  Injector Pump ", "Fuel System 1  Service Pump ", "Cooling System 1 Service Pump "]
    sensors["s2_g3"] = ["Fuel System 2  Injector Pump ", "Fuel System 2  Service Pump ", "Cooling System 2 Service Pump "]
    sensors["combined_g3"] = sensors["s1_g3"] + sensors["s2_g3"]

    return sensors

# Loads data and places them into PyTorch Dataset and Dataloader objects for all sensor groups
def get_data(data_dir, sequences_filename, num_iters=1, batch_size=4):
    data_dict = parse_avg_sensor_data(data_dir, sequences_filename)

    sensor_groups = create_sensor_groups()

    sensor_to_datasets = {}

    for group_id, sensor_group in sensor_groups.items():
        datasets, dataloaders = create_train_test_val_splits(data_dict, sensor_group, [0.8, 0.1, 0.1], num_iters=num_iters, batch_size=batch_size)
        sensor_to_datasets[group_id] = (datasets, dataloaders)

    return sensor_to_datasets

# Saves datasets and dataloaders to files
def save_data(sensor_to_datasets, save_dir):
    
    for sensor_group, t in sensor_to_datasets.items():
        datasets, dataloaders = t

        # Create dir
        dir = save_dir + "/" + sensor_group
        os.makedirs(dir, exist_ok=True)

        for op_prof in range(1,4):
            op_dir = dir + "/op_prof_" + str(op_prof)
            os.makedirs(op_dir, exist_ok=True)

            # Save datasets to dir
            seq_num = 4
            for dataset in datasets[op_prof]:
                torch.save(dataset, op_dir + "/" + "dataset_seq_" + str(seq_num) + ".pt")
                seq_num += 1

# Performs a 2nd degree polynomial interpolation on the specified dataset
# Returns the interpolated dataset
def polynomial_interpolation(dataset:RULDataset, num_points:int, sensor_group:list, stdv:dict, sequence_length:int=None) -> RULDataset:
    sensor_data = dataset.sensor_data
    N, T, S = sensor_data.shape

    interpolated_data = np.zeros((N, num_points, S))

    for i in range(N):
        for j in range(S):
            x = np.arange(T)
            y = sensor_data[i, :, j]
            f = sp.interpolate.interp1d(x, y, kind="quadratic")
            x_new = np.linspace(0, T-1, num_points)
            y_new = f(x_new)

            # Apply noise
            # NOTE the standard deviations seem to be extremely large and move away from the spirit of the data. So, for now I will not apply noise. This will be revisited
            # noise = np.random.normal(0, stdv[sensor_group[j]], num_points)
            # y_new += noise

            interpolated_data[i, :, j] = y_new
            interpolated_data[i, 0, j] = sensor_data[i, 0, j]
            interpolated_data[i, -1, j] = sensor_data[i, -1, j]   

    # Don't think I need to adjust the labels
    return RULDataset(interpolated_data, dataset.rul_labels, sequence_length)

# Interpolates and applies noise to all datasets
# Returns new sensor_to_datasets dict
def apply_polynomial_interpolation(sensor_to_datasets:dict, sensor_groups:dict, num_points:int, stdv:dict, sequence_length:int=None) -> dict:
    out_sensor_to_datasets = {}

    for sensor_group_name, t in sensor_to_datasets.items():
        datasets, dataloaders = t
        new_datasets = {}
        for op_prof in range(1,4):
            new_datasets[op_prof] = []
            for dataset in datasets[op_prof]:
                new_datasets[op_prof].append(polynomial_interpolation(dataset, num_points, sensor_groups[sensor_group_name], stdv, sequence_length))

        out_sensor_to_datasets[sensor_group_name] = (new_datasets, dataloaders)

    return out_sensor_to_datasets

# Calculates standard deviation for each sensor in the specified sensor group from the dataset
# where df is a DataFrame of a run from a failure profile
def calc_stdv(sensor_group:list, df:pd.DataFrame) -> dict:
    stdv = {}
    for sensor in sensor_group:
        stdv[sensor] = df[sensor]

    return stdv

# Creates a DataFrame from the output processed files of a particular failure profile and operational profile
def create_df_from_proc_data(dir:str) -> pd.DataFrame:
    
    # Flow
    df_flow = pd.read_csv(dir + "/out_CRIO1_Flow_data_Profile_1_Run_4_Operation_1.csv")
    flow_std = df_flow.std(axis=0)

    # Main
    df_main1 = pd.read_csv(dir + "/out_CRIO1_mAin_Profile_1_Run_4_Operation_1.csv")
    df_main2 = pd.read_csv(dir + "/out_CRIO2_mAin_Profile_1_Run_4_Operation_1.csv")
    main1_std = df_main1.std(axis=0)
    main2_std = df_main2.std(axis=0)

    # Vin
    df_vin1 = pd.read_csv(dir + "/out_CRIO1_Vin_Profile_1_Run_4_Operation_1.csv")
    df_vin2 = pd.read_csv(dir + "/out_CRIO2_Vin_Profile_1_Run_4_Operation_1.csv")
    vin1_std = df_vin1.std(axis=0)
    vin2_std = df_vin2.std(axis=0)

    df_out = pd.concat([flow_std, main1_std, main2_std, vin1_std, vin2_std], axis=0)

    return df_out

# Calculate the tolerance for the voting procedure
def calc_tolerance(v1, v2):
    return abs(v1 - v2) / v1



# Based on the voting procedure outlined in Andy's thesis, determine the overall fueling profile
# Additionally, fix the temperature sensor data anomalies
# Calculate specific sensors (delta pressure, delta temp in cooling system)
def correct_data(data_dir:str, save_loc:str, sensor_group:list):

    # Walk through the directory
    for i in range(1, 101):
        fp_dir = data_dir + "/Failure_Profile_" + str(i)

        # Fix temperature anomalies and calc delta temp and delta pressure
        op_prof_to_dt = {}
        op_prof_to_dp = {}
        for j in range(1,4):
            filename = "AverageValueData_Failure_Profile_" + str(i) + "_Operational_Profile_" + str(j) + ".csv"
            fp_df = pd.read_csv(fp_dir + "/" + filename)

            # Fix temperature sensor data anomalies and calc deltas
            dt1 = []
            dt2 = []
            dp1 = []
            dp2 = []
            for idx, row in fp_df.iterrows():

                # Calc delta pressure
                delta_p_1 = row["Cooling System 1 P 1"] - row["Cooling System 1 P 2"]
                delta_p_2 = row["Cooling System 2 P 1"] - row["Cooling System 2 P 2"]
                dp1.append(delta_p_1)
                dp2.append(delta_p_2)

                # Calc delta temp and fix anomalies
                delta_t_1 = row["Cooling System 1 Heater Output Temperature"] - row["Cooling System 1 Heater Input Temperature "]
                delta_t_2 = row["Cooling System 2  Heater Output Temperature"] - row["Cooling System 2 Heater Input Temperature "]

                if idx == 0:
                    dt1.append(delta_t_1)
                    dt2.append(delta_t_2)
                    continue

                if delta_t_1 > 10 or delta_t_1 < 0:
                    delta_t_1 = dt1[-1]
                if delta_t_2 > 10 or delta_t_2 < 0:
                    delta_t_2 = dt2[-1]

                dt1.append(delta_t_1)
                dt2.append(delta_t_2)

            op_prof_to_dt[j] = (dt1, dt2)
            op_prof_to_dp[j] = (dp1, dp2)
        
            
        # Fix fueling system data anomalies
        df1 = pd.read_csv(fp_dir + "/AverageValueData_Failure_Profile_" + str(i) + "_Operational_Profile_1.csv")
        df2 = pd.read_csv(fp_dir + "/AverageValueData_Failure_Profile_" + str(i) + "_Operational_Profile_2.csv")
        df3 = pd.read_csv(fp_dir + "/AverageValueData_Failure_Profile_" + str(i) + "_Operational_Profile_3.csv")

        dfs = [df1, df2, df3]

        fueling_sensor_list = ["Fuel System 1  LP 1 ", "Fuel System 2  LP 1", "Fuel System 1 High Pressure Rail", "Fuel System 2  High Pressure Rail", "Fuel System 1  Injector Pump ", "Fuel System 2  Injector Pump ", "Fuel System 1  Service Pump ", "Fuel System 2  Service Pump ", "Fuel System 1  HP Relief Flow", "Fuel System 2  HP Relief Flow", "Fuel System 1  Injector Flow", "Fuel System 2  Injector Flow", "Fuel System 1  Service Flow", "Fuel System 2  Service Flow"]
        out_fuel_df = pd.DataFrame(columns=fueling_sensor_list)
        fuel_data_dict = {sensor:[] for sensor in fueling_sensor_list}
        for idx in range(len(df1)):
            
            for sensor in fueling_sensor_list:
            
                # Get each value from df
                val1 = df1.iloc[idx][sensor]
                val2 = df2.iloc[idx][sensor]
                val3 = df3.iloc[idx][sensor]

                vals = [val1, val2, val3]

                # Voting procedure
                j = np.argmin([calc_tolerance(val1, val2), calc_tolerance(val1, val3), calc_tolerance(val2, val3)])
                if j == 2:
                    fuel_data_dict[sensor].append(vals[1])
                else:
                    fuel_data_dict[sensor].append(vals[0])

        for sensor in fueling_sensor_list:
            out_fuel_df[sensor] = fuel_data_dict[sensor]

        # Put everything into dataframes for each operational profile and save
        for j in range(1,4):
            op_dir = save_loc + "/Failure_Profile_" + str(i)
            os.makedirs(op_dir, exist_ok=True)

            # Get other sensors
            columns = fueling_sensor_list + ["Cooling System 1 Delta Temp", "Cooling System 2 Delta Temp", "Cooling System 1 Delta Pressure", "Cooling System 2 Delta Pressure"] + ["Cooling System 1 Flow", "Cooling System 2 Flow", "Cooling System 1 Service Pump ", "Cooling System 2 Service Pump "]
            out_df = pd.DataFrame(columns=columns)

            # Fueling data
            for sensor in fueling_sensor_list:
                out_df[sensor] = out_fuel_df[sensor]

            # Delta temp and delta pressure
            out_df["Cooling System 1 Delta Temp"] = op_prof_to_dt[j][0]
            out_df["Cooling System 2 Delta Temp"] = op_prof_to_dt[j][1]
            out_df["Cooling System 1 Delta Pressure"] = op_prof_to_dp[j][0]
            out_df["Cooling System 2 Delta Pressure"] = op_prof_to_dp[j][1]

            # Rest of cooling system data
            out_df["Cooling System 1 Flow"] = dfs[j-1]["Cooling System 1 Flow"]
            out_df["Cooling System 2 Flow"] = dfs[j-1]["Cooling System 2 Flow"]
            out_df["Cooling System 1 Service Pump "] = dfs[j-1]["Cooling System 1 Service Pump "]
            out_df["Cooling System 2 Service Pump "] = dfs[j-1]["Cooling System 2 Service Pump "]

            # Save df
            out_df.to_csv(op_dir + "/AverageValueData_Failure_Profile_" + str(i) + "_Operational_Profile_" + str(j) + ".csv", index=False)



if __name__ == "__main__":

    data_dir = "data/AvgValue_Data"
    sequences_filename = "data/Failure_Profile_Labels/labels.csv"

    # stdv_df = create_df_from_proc_data("data/Operational_Profile_1/")

    # data_dict = parse_avg_sensor_data(data_dir, sequences_filename)
    sensor_groups = create_sensor_groups()

    combined = sensor_groups["combined_g1"]
    # stdv = calc_stdv(combined, stdv_df)

    # Fix data
    correct_data(data_dir, "data/Corrected_AvgValue_Data", combined)

    # sensors_to_datasets = get_data(data_dir, sequences_filename, num_iters=1, batch_size=4)
    # save_data(sensors_to_datasets, "data/processed_data/original")

    # Interpolate and apply noise
    # num_points = 30
    # sensors_to_datasets = apply_polynomial_interpolation(sensors_to_datasets, sensor_groups, num_points, stdv, sequence_length=3)
    # save_data(sensors_to_datasets, "data/processed_data/interpolated/" + str(num_points))