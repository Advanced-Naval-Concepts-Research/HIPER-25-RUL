import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


def plot_loss_acc(dir:str, regular:bool=True):

    models = ["Base", "LSTM", "LSTMCNN"]

    partitions = ["A", "B", "C"]
    op_profs = ["1", "2", "3"]
    sensors = ["s1_g1", "s1_g2", "s1_g3"]

    out_df = pd.DataFrame(columns=["model", "partition", "regular", "op_prof", "sensor", "seq_len", "loss", "acc"])

    # Want to show
    # 1. Compare loss and accuracy with x=sequence, y=loss/acc, labels=op_prof, don't care about partition
        # This will show how the model performs with different sequence lengths
        # Repeat for each model
    # 2. Same as (1) but compare across partitions rather than op_prof
        # This will show how the model performs with different partitions
        # Repeat for each model

    # Plot based on sensor groupings across op_prof
    # Repeat for each sequence length
    for model in models:

        partition_loss_avg = {4:{"A":[], "B":[], "C":[]}, 5:{"A":[], "B":[], "C":[]}, 6:{"A":[], "B":[], "C":[]}}
        partition_acc_avg = {4:{"A":[], "B":[], "C":[]}, 5:{"A":[], "B":[], "C":[]}, 6:{"A":[], "B":[], "C":[]}}


        for partition in partitions:
            for sensor in sensors:
                for op_prof in op_profs:

                    if regular:
                        file_loss = dir + "/" + model + "/" + partition + "/regular/" + sensor + "/" + op_prof + "/loss.csv"
                        file_acc = dir + "/" + model + "/" + partition + "/regular/" + sensor + "/" + op_prof + "/accuracy.csv"
                    else:
                        file_loss = dir + "/" + model + "/" + partition + "/minmax/" + sensor + "/" + op_prof + "/loss.csv"
                        file_acc = dir + "/" + model + "/" + partition + "/minmax/" + sensor + "/" + op_prof + "/accuracy.csv"

                    loss = pd.read_csv(file_loss)
                    acc = pd.read_csv(file_acc)

                    # Calc averages
                    for seq in [4, 5, 6]:
                        partition_loss_avg[seq][partition].append(loss[str(seq)].mean())
                        partition_acc_avg[seq][partition].append(acc[str(seq)].mean())


                    # Add to dataframe
                    for seq in [4, 5, 6]:
                        out_df.loc[len(out_df), out_df.columns] = model, partition, "regular", op_prof, sensor, seq, loss[str(seq)].mean(), acc[str(seq)].mean()

                    # Add data for other
                    if regular:
                        file_loss = dir + "/" + model + "/" + partition + "/minmax/" + sensor + "/" + op_prof + "/loss.csv"
                        file_acc = dir + "/" + model + "/" + partition + "/minmax/" + sensor + "/" + op_prof + "/accuracy.csv"
                    else:
                        file_loss = dir + "/" + model + "/" + partition + "/regular/" + sensor + "/" + op_prof + "/loss.csv"
                        file_acc = dir + "/" + model + "/" + partition + "/regular/" + sensor + "/" + op_prof + "/accuracy.csv"

                    loss = pd.read_csv(file_loss)
                    acc = pd.read_csv(file_acc)

                    # Add to dataframe
                    for seq in [4, 5, 6]:
                        out_df.loc[len(out_df), out_df.columns] = model, partition, "minmax", op_prof, sensor, seq, loss[str(seq)].mean(), acc[str(seq)].mean()



        # Plot loss and bin
        seq_loss_temp = {"A":[], "B":[], "C":[]}
        for seq in [4,5,6]:
            seq_loss_temp["A"] += partition_loss_avg[seq]["A"]
            seq_loss_temp["B"] += partition_loss_avg[seq]["B"]
            seq_loss_temp["C"] += partition_loss_avg[seq]["C"]
        
        plt.scatter([4]*len(seq_loss_temp["A"]), seq_loss_temp["A"], label="A")
        plt.scatter([5]*len(seq_loss_temp["B"]), seq_loss_temp["B"], label="B")
        plt.scatter([6]*len(seq_loss_temp["C"]), seq_loss_temp["C"], label="C")

        plt.legend()
        plt.xticks([4, 5, 6])
        plt.xlabel("Sequence Length")
        plt.ylabel("Loss")
        plt.title(model + " Loss")

        save_dir = dir + "/figs/" + model + "_loss.png"

        plt.savefig(save_dir)
        plt.close()

        # Plot acc and bin
        seq_acc_temp = {"A":[], "B":[], "C":[]}
        for seq in [4,5,6]:
            seq_acc_temp["A"] += partition_acc_avg[seq]["A"]
            seq_acc_temp["B"] += partition_acc_avg[seq]["B"]
            seq_acc_temp["C"] += partition_acc_avg[seq]["C"]
        
        plt.scatter([4]*len(seq_acc_temp["A"]), seq_acc_temp["A"], label="A")
        plt.scatter([5]*len(seq_acc_temp["B"]), seq_acc_temp["B"], label="B")
        plt.scatter([6]*len(seq_acc_temp["C"]), seq_acc_temp["C"], label="C")

        # plt.legend()
        # plt.xticks([4, 5, 6])
        # plt.xlabel("Sequence Length")
        # plt.ylabel("Accuracy")
        # plt.title(model + " Accuracy")

        # save_dir = dir + "/figs/" + model + "_acc.png"

        # plt.savefig(save_dir)
        # plt.close()

    out_df.to_csv(dir + "/figs/stats.csv", index=False)

    return out_df

if __name__ == "__main__":

    df = plot_loss_acc("data/stats", regular=True)
    pass