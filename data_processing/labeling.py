import numpy as np
import pandas as pd

def label_data(dir, leaks):

    # Limits
    # Leaks occur between 1-4, otherwise, they are clogging failures
    cool_lim = 1
    cool_temp_lim = 3
    fuel_lp_lim = 0.5
    fuel_hp_lim = 80

    labels = {i:[] for i in range(1,101)}

    # Cycle through all failure profiles and determine when each failure profile failed

    for i in range(1, 101):
        if i in leaks:
            continue
        fp_dir = dir + "/Failure_Profile_" + str(i)

        # Read in failure profiles for each operational profile
        for j in range(1,4):
            filename = "AverageValueData_Failure_Profile_" + str(i) + "_Operational_Profile_" + str(j) + ".csv"
            fp_df = pd.read_csv(fp_dir + "/" + filename)
            
            # Determine when the failure profile failed
            for k in range(1, len(fp_df) + 1):
                row = fp_df.iloc[k - 1]
                if row["Cooling System 1 Flow"] < cool_lim or row["Cooling System 2 Flow"] < cool_lim:
                    labels[i].append(k)
                    break
                elif row["Cooling System 1 Delta Temp"] > cool_temp_lim or row["Cooling System 2 Delta Temp"] > cool_temp_lim:
                    labels[i].append(k)
                    break
                # elif row["Fuel System 1  LP 1 "] < fuel_lp_lim or row["Fuel System 2  LP 1"] < fuel_lp_lim:
                #     labels[i].append(k)
                #     break
                # elif row["Fuel System 1 High Pressure Rail"] < fuel_hp_lim or row["Fuel System 2  High Pressure Rail"] < fuel_hp_lim:
                #     labels[i].append(k)
                #     break
            else:
                # labels[i].append(len(fp_df) + 1)
                labels[i].append(100)


    print(labels)
    label_df = pd.DataFrame(columns=["Failure Profile", "Profile A", "Profile B", "Profile C"])
    for fp, seq in labels.items():
        if len(seq) == 0:
            continue
        label_df.loc[len(label_df)] = [fp, seq[0], seq[1], seq[2]]
    label_df.to_csv("data/Failure_Profile_Labels/labels_cooling.csv", index=False)
    # leak_count = 0
    # for fp in labels:
    #     if (np.array(fp) < 6).any():
    #         leak_count += 1
    # print(leak_count)

def check_leaks(dir):
    # Examines simulation control for leaks

    leak_fps = []
    leaks_cool = []
    leaks_fuel = []
    for i in range(1, 101):
        file = dir + "/Simulation_Control" + str(i) + ".csv"
        df = pd.read_csv(file)
        for j in range(len(df)):
            row = df.iloc[j]
            if row["C1_Clog_LeakTime"] == 1 or row["C2_Clog_LeakTime"] == 1:
                # print("Leak in cooling system")
                leak_fps.append(i)
                leaks_cool.append(i)
            if row["F1_HP_Leak_Time"] == 1 or row["F2_HP_Clog_LeakTime"] == 1:
                # print("Leak in HP fuel system")
                leak_fps.append(i)
                leaks_fuel.append(i)
            if row["F1_LP_Clog_LeakTime"] == 1 or row["F2_LP_Clog_LeakTime"] == 1:
                # print("Leak in LP fuel system")
                leak_fps.append(i)
                leaks_fuel.append(i)

    leak_fps = list(set(leak_fps))
    leaks_cool = list(set(leaks_cool))
    leaks_fuel = list(set(leaks_fuel))
    print(leak_fps)
    return leak_fps, leaks_cool, leaks_fuel

if __name__ == "__main__":
    leaks, leaks_cool, leaks_fuel = check_leaks("data/SimulationControl/")
    # label_data("data/Corrected_AvgValue_Data/", leaks)
    not_leaks = [i-1 for i in range(1,101) if i not in leaks]
    leaks_idx = [i-1 for i in leaks]
    leaks_fuel_idx = [i-1 for i in leaks_fuel]
    leaks_cool_idx = [i-1 for i in leaks_cool]

    not_leaks_fuel_idx = [i-1 for i in range(1,101) if i not in leaks_fuel]
    not_leaks_cool_idx = [i-1 for i in range(1,101) if i not in leaks_cool]

    cool_df = pd.read_excel("data/Failure_Profile_Labels/incorrect labels/cooling_labels.xlsx")
    fuel_df = pd.read_excel("data/Failure_Profile_Labels/incorrect labels/fuel_labels.xlsx")

    comb_df = pd.DataFrame(columns=["Failure Profile", "Profile A", "Profile B", "Profile C"])

    # Probable fuel labels
    fuel_labels = fuel_df["Runs"].iloc[not_leaks_fuel_idx]

    # Probable cooling labels
    cool_labels_a = cool_df["a"].iloc[not_leaks_cool_idx] #not sure if just second plant or both. Only seem to be different by 1 or 2 numbers at most. Using first plant for now since counts for "B" and "C" match first plant numbers
    cool_labels_b = cool_df["b"].iloc[not_leaks_cool_idx]
    cool_labels_c = cool_df["c"].iloc[not_leaks_cool_idx]

    # Just do plant 1 for now
    
    
    # Combining labels
    print("here")
    comb_df["Failure Profile"] = list(range(1,101))
    temp_df = pd.DataFrame(columns=["Failure Profile", "A", "B", "C", "Fuel"])
    temp_df["Failure Profile"] = list(range(1,101))
    temp_df["A"] = cool_df["a"]
    temp_df["B"] = cool_df["b"]
    temp_df["C"] = cool_df["c"]
    temp_df["Fuel"] = fuel_df["Runs"]

    temp_df.iloc[leaks_idx] = -1


    comb_df["Profile A"] = temp_df[["A", "Fuel"]].min(axis=1)
    comb_df["Profile B"] = temp_df[["B", "Fuel"]].min(axis=1)
    comb_df["Profile C"] = temp_df[["C", "Fuel"]].min(axis=1)

    comb_df.to_csv("data/Failure_Profile_Labels/labels_combined.csv", index=False)