import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCNNModel1(nn.Module):
    def __init__(self, input_size:int, lstm_hidden_size1:int=21, lstm_out_size2:int = 10):
        super(LSTMCNNModel1, self).__init__()
    
        # LSTM component
        self.lstm1 = nn.LSTM(input_size, lstm_hidden_size1, batch_first=True)  # First LSTM layer
        
        # this LSTM is the one the glues everything along with fc layer
        self.lstm2 = nn.LSTM(lstm_hidden_size1, lstm_out_size2, batch_first=True) 
        self.fc = nn.Linear(lstm_out_size2, 1) # 1 class to connect to, the RUL prediction
        
        
        # CNN component
        
        #TODO PARAMS
         # 1) Convolution 1: TBD PARAMS
        self.conv1 = nn.Conv2d(
            in_channels=1, # data is 2D at the start
            out_channels=3, 
            kernel_size=(3,2), 
            padding=(1,0)  # padding vertically
        )
        
        #TODO PARAMS
        # MaxPooling layer TBD params
        self.pool = nn.MaxPool2d(kernel_size=2, stride=(2,2))
        

    def forward(self, x):
        #x Shape is (N, T, 3, 14)
        N = x.shape[0]
        T = x.shape[1]
        x1 = x.copy()
        # branch 1 (first LSTM): 
            # flatten out the 2D data of the timestep
            # run it into LSTM1
            # get its output lstm_out
        lsmtout = x1.reshape(x1.shape[0],x1.shape[1], -1)
        lstmout = self.lstm1(lstmout)
        # branch 2 (CNN)
            # convolve once
            # pool
            # get its output CNN_out
        # from here, dims of lstm_out and CNN_out should match. We will add their results
        
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
        cnnout =  self.conv1(x.unsqueeze(1))
        #unsqueeze(1) to allow num channels
        cnnout = self.pool(cnnout)
        #flatten
        cnnout = self.reshape(N, T, -1)
        
        assert lstmout.shape == cnnout.shape, "Shapes dont match in CNNLSTMDAG"
        
        # then, throw lstm_out + CNN_out (elementwise addition) into 2nd LSTM
        out = self.lstm2(lstmout + cnnout)
        # grab last timestep
        out = out[:,-1,:]
        
        # Take that output and fully connect its layer into the RUL prediction
        return self.fc(out)
        
        
