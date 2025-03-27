# RUL Model from Stephen A. Olson's Thesis
# Taken from Table 5.20 in his thesis

# TODO what does Andy do with Z score normalization?

import torch
import torch.nn as nn

class BaseRULModel(nn.Module):
    def __init__(self, input_size):
        super(BaseRULModel, self).__init__()
        self.hidden_size = 200
        self.num_layers = 1 
        self.num_classes = 1
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out, (hn, cn) = self.lstm(x) Not sure which is correct. Both seem to work

        out = self.fc(out[:, -1, :])
        return out
    
    def get_name(self):
        return "BaseRULModel"