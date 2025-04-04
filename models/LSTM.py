# LSTM Model from Reliable Prediction of Remaining Useful Life for AircraftEngines: An LSTM-Based Approach with Conservative LossFunctionAnees Peringal∗, Mohammed Basheer Mohiuddin†, Abdel Gafoor Haddad‡ and Praveen Kumar Muthusamy§Khalifa University, Abu Dhabi, UAE
# https://github.com/AneesPeringal/rul-prediction/blob/main/rul_prediction.py

import torch
import torch.nn as nn
import torch.nn.functional as functional

class LSTMModel(nn.Module):
    def __init__(self, input_size, sequence_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = 200
        self.num_layers = 2

        self.lstm1 = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.0)
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear1 = nn.Linear(self.hidden_size * sequence_size, 100)
        self.dropout2 = nn.Dropout(p=0.0)
        self.linear2 = nn.Linear(100,1)

    
    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,device=x.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,device=x.device).requires_grad_()

        out, (hn, cn) = self.lstm1(x, (h0, c0))
        out = self.dropout1(out)
        out, (hn, cn) = self.lstm2(out, (h0, c0))
        out = out.flatten(start_dim=1)
        out = self.linear1(out)
        out = self.dropout2(out)
        out = self.linear2(out)
        
        return out
    
    def get_name(self):
        return "LSTM"