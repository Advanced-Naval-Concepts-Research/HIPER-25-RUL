import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        
        # Layer sizes, dropout rates, and L2 regularization coefficient
        layer_sizes = [484, 504, 280, 180, 89, 50, 29]
        dropout_rates = [0.59, 0.59, 0.589, 0.591, 0.59, 0.59]
        l2_reg = 0.001

        layers = []
        
        # Create layers based on given sizes and dropout rates
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())  # ReLU activation
            layers.append(nn.Dropout(dropout_rates[i]))  # Dropout layer
            
        layers.append(nn.Linear(layer_sizes[len(layer_sizes)-1], 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize the L2 regularization coefficient (it will be added during the optimization step)
        self.l2_reg = l2_reg

    def forward(self, x):
        return self.network(x)


class LSTMCNNModel(nn.Module):
    def __init__(self, input_size:int, lstm_hidden_size1:int=100, lstm_out_size2:int = 4, lstm_dropout_rate:float = 0.2):
        super(LSTMCNNModel, self).__init__()
        
        #DNN made in another class
        self.DNN1 = DNN()
        # LSTM component
        
        self.lstm1 = nn.LSTM(input_size, lstm_hidden_size1, batch_first=True)  # First LSTM layer
        self.dropout1 = nn.Dropout(lstm_dropout_rate)  # dropout rate was not specified by paper. 0.2 is used
        
        self.lstm2 = nn.LSTM(lstm_hidden_size1, lstm_out_size2, batch_first=True)  # Out size is 4 implied by paper
        # althought it was not directly said
        self.dropout2 = nn.Dropout(lstm_dropout_rate)  # dropout rate was not specified by paper. 0.2 is used
        
        # CNN component
        
         # 1) Convolution 1: kernel_size=3
        self.conv1 = nn.Conv2d(
            in_channels=1, # data is 2D at the start
            out_channels=30, 
            kernel_size=3, 
            padding=1  # padding=1 to preserve spatial size for a 3×3 kernel
        )
        
        # 2) Convolution 2: kernel_size=3
        self.conv2 = nn.Conv2d(
            in_channels=30, 
            out_channels=60, 
            kernel_size=3, 
            padding=1
        )
        
        # 3) Convolution 3: kernel_size=2
        self.conv3 = nn.Conv2d(
            in_channels=60, 
            out_channels=120, 
            kernel_size=2, 
            padding=0
        )
        # MaxPooling layer (2×2), used twice
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # DNN to glue

    def forward(self, x):
        lstm_out, _ = self.lstm1(x)  # Output shape: (batch_size, timesteps, hidden_size1)
        lstm_out = self.dropout1(lstm_out)
        
        lstm_out, _ = self.lstm2(lstm_out)  # Output shape: (batch_size, timesteps, hidden_size2)
        lstm_out = self.dropout2(lstm_out)
        lstm_out = lstm_out[:, -1, :]  # Taking the last timestep output
        
        # paper is super unclear on this. They do the CNN over the correlation matrix I believe though
        mean_x = x.mean(dim=0, keepdim=True)  # Mean over features (dim=0)
        std_x = x.std(dim=0, keepdim=True, unbiased=False)  # Standard deviation over features
        # Normalize the input data (standardization)
        normalized_x = (x - mean_x) / (std_x + 1e-6)  # Avoid division by zero
        # Compute the correlation matrix (Pearson correlation) for 14 samples
        correlation_matrix = torch.mm(normalized_x, normalized_x.T) / (x.shape[0] - 1)
        
        # [Conv -> ReLU -> MaxPool]
        cnn_out = F.relu(self.conv1(correlation_matrix))
        cnn_out = self.pool(cnn_out)
        
        # [Conv -> ReLU -> MaxPool]
        cnn_out = F.relu(self.conv2(cnn_out))
        cnn_out = self.pool(cnn_out)
        
        # [Conv -> ReLU]
        cnn_out = F.relu(self.conv3(cnn_out))
        
        # Flatten
        cnn_out = torch.flatten(cnn_out, start_dim=1)  # Flatten all but batch dimension
        
        combined = torch.cat((lstm_out, cnn_out), dim=1)
        assert combined.shape[1] == 484, f"Expected 484, but got {combined.shape[1]}"
        
        self.DNN1.forward(combined)

