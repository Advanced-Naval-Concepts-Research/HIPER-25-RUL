import torch
import torch.nn as nn
import torch.optim as optim

class OvercompleteAutoencoder(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=50):
        super(OvercompleteAutoencoder, self).__init__()
        
        # --------------------
        #     ENCODER
        # --------------------
        # Layer 1: 21 -> 21
        # Layer 2: 21 -> 50
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # --------------------
        #     DECODER
        # --------------------
        # Layer 1: 50 -> 21
        # Layer 2: 21 -> 21
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(), # sigmoid to make it nontrivial for the NN to reconstruct the data accurately
            nn.Linear(input_dim, input_dim),
        )
    
    def forward(self, x):
        # Encode to latent space
        latent = self.encoder(x)
        # Decode (reconstruct) back to original dimension
        reconstructed = self.decoder(latent)
        return reconstructed