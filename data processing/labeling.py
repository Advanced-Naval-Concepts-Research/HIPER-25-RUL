import numpy as np
import pandas as pd

def label_data(dir, leaks):

    # Limits
    # Leaks occur between 1-4, otherwise, they are clogging failures
    cool_lim = 1
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
                elif row["Fuel System 1  LP 1 "] < fuel_lp_lim or row["Fuel System 2  LP 1"] < fuel_lp_lim:
                    labels[i].append(k)
                    break
                elif row["Fuel System 1 High Pressure Rail"] < fuel_hp_lim or row["Fuel System 2  High Pressure Rail"] < fuel_hp_lim:
                    labels[i].append(k)
                    break
            else:
                labels[i].append(len(fp_df))

    print(labels)
    label_df = pd.DataFrame(columns=["Failure Profile", "Profile A", "Profile B", "Profile C"])
    for fp, seq in labels.items():
        if len(seq) == 0:
            continue
        label_df.loc[len(label_df)] = [fp, seq[0], seq[1], seq[2]]
    label_df.to_csv("data/Failure_Profile_Labels/labels.csv", index=False)
    # leak_count = 0
    # for fp in labels:
    #     if (np.array(fp) < 6).any():
    #         leak_count += 1
    # print(leak_count)

def check_leaks(dir):
    # Examines simulation control for leaks

    leak_fps = []
    for i in range(1, 101):
        file = dir + "/Simulation_Control" + str(i) + ".csv"
        df = pd.read_csv(file)
        for j in range(len(df)):
            row = df.iloc[j]
            found = False
            if row["C1_Clog_LeakTime"] == 1 or row["C2_Clog_LeakTime"] == 1:
                print("Leak in cooling system")
                if not found:
                    leak_fps.append(i)
                    found = True
            if row["F1_HP_Leak_Time"] == 1 or row["F2_HP_Clog_LeakTime"] == 1:
                print("Leak in HP fuel system")
                if not found:
                    leak_fps.append(i)
                    found = True
            if row["F1_LP_Clog_LeakTime"] == 1 or row["F2_LP_Clog_LeakTime"] == 1:
                print("Leak in LP fuel system")
                if not found:
                    leak_fps.append(i)
                    found = True

            if found:
                break
    print(leak_fps)
    return leak_fps

if __name__ == "__main__":
    leaks = check_leaks("data/SimulationControl/")
    label_data("data/AvgValue_Data/", leaks)
