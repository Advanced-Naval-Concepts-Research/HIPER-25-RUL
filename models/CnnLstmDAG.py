import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCNNModel1(nn.Module):
    def __init__(self, input_size:int, lstm_hidden_size1:int=100, lstm_out_size2:int = 4, lstm_dropout_rate:float = 0.2):
        super(LSTMCNNModel1, self).__init__()
    
        # LSTM component
        self.lstm1 = nn.LSTM(input_size, lstm_hidden_size1, batch_first=True)  # First LSTM layer
        
        # this LSTM is the one the glues everything along with fc layer
        self.lstm2 = nn.LSTM(lstm_hidden_size1, lstm_out_size2, batch_first=True)  # Out size is 4 implied by paper
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        
        # CNN component
        
        #TODO PARAMS
         # 1) Convolution 1: TBD PARAMS
        self.conv1 = nn.Conv2d(
            in_channels=1, # data is 2D at the start
            out_channels=30, 
            kernel_size=3, 
            padding=1  # padding=1 to preserve spatial size for a 3×3 kernel
        )
        
        #TODO PARAMS
        # MaxPooling layer TBD params
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # DNN to glue

    def forward(self, x):
        # branch 1 (first LSTM): 
            # flatten out the 2D data of the timestep
            # run it into LSTM1
            # get its output lstm_out
        # branch 2 (CNN)
            # convolve once
            # pool
            # get its output CNN_out
        # from here, dims of lstm_out and CNN_out should match. We will add their results
        
        # then, throw lstm_out + CNN_out (elementwise addition) into 2nd LSTM
        
        # Take that output and fully connect its layer into the RUL prediction
        
