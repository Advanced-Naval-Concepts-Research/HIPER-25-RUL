import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCNNModel1(nn.Module):
    def __init__(self, input_size:int, lstm_out_size2:int = 10):
        super(LSTMCNNModel1, self).__init__()
    
        # data dim is (3, input_size)
        # after conv it is (3, 2, input_size)
        # granting us a final result after pooling (3,1,floor(input_size/2))
        # LSTM component
        lstm_hidden_size1 = 3 * int(input_size/2)
        self.lstm1 = nn.LSTM(input_size * 3, lstm_hidden_size1, batch_first=True)  # First LSTM layer
        
        # this LSTM is the one the glues everything along with fc layer
        self.lstm2 = nn.LSTM(lstm_hidden_size1, lstm_out_size2, batch_first=True) 
        self.fc = nn.Linear(lstm_out_size2, 1) # 1 class to connect to, the RUL prediction
        
        
        # CNN component
        
        #TODO PARAMS
         # 1) Convolution 1: TBD PARAMS
        self.conv1 = nn.Conv2d(
            in_channels=1, # data is 2D at the start
            out_channels=3, 
            kernel_size=(2,3), 
            padding=(0,1)  # padding vertically
        )
        
        #TODO PARAMS
        # MaxPooling layer TBD params
        self.pool = nn.MaxPool2d(kernel_size=2, stride=(2,2))
        

    def forward(self, x):
        #x Shape is (N, T, 3, 14)
        N = x.shape[0]
        T = x.shape[1]
        x1 = x.detach().clone()
        # branch 1 (first LSTM): 
            # flatten out the 2D data of the timestep
            # run it into LSTM1
            # get its output lstm_out
        lstmout = x1.reshape(x1.shape[0],x1.shape[1], -1)
        lstmout = self.lstm1(lstmout)[0]
       
       
        # branch 2 (CNN)
            # convolve once
            # pool
            # get its output CNN_out
        # from here, dims of lstm_out and CNN_out should match. We will add their results
       
        xx = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
        xx = xx.unsqueeze(1)
        cnnout =  self.conv1(xx)
        #unsqueeze(1) to allow num channels
        cnnout = self.pool(cnnout)
        #flatten
        cnnout = cnnout.reshape(N, T, -1)
        assert lstmout.shape == cnnout.shape, f"Shapes dont match in CNNLSTMDAG x:{x.shape} lstm:{lstmout.shape}, cnn:{cnnout.shape}"
        
        # then, throw lstm_out + CNN_out (elementwise addition) into 2nd LSTM
        out = self.lstm2(lstmout + cnnout)[0]
        # grab last timestep
        out = out[:,-1,:]
        
        # Take that output and fully connect its layer into the RUL prediction
        return self.fc(out)

    def get_name(self):
        return "LSTMCNN"
        
