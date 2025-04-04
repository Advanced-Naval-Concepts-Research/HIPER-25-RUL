import torch
from models.overcomplete_autoencoder import OvercompleteAutoencoder
from data_processing.data_preprocessing import create_sensor_groups
from main import load_dataset
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from data_processing.data_preprocessing import RULDataset as RULDataset
import os
def generate_latent_representations(model, dataloader, device='cpu'):
    """
    Generates latent representations using the encoder of the autoencoder
    
    Args:
        model (OvercompleteAutoencoder): Trained autoencoder model
        dataloader (DataLoader): Dataloader providing input data batches
        device (str): Device to run computations on ('cpu' or 'cuda')
    
    Returns:
        torch.Tensor: Tensor containing all generated latent representations
    """
    model.eval()  # Set model to evaluation mode
    model.to(device)
    
    latent_vectors = []
    
    with torch.no_grad():
        for batch in dataloader:
           
            # Extract inputs - handle different dataloader formats
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            A,B,C = inputs.shape[0], inputs.shape[1], inputs.shape[2]
            inputs = inputs.reshape(A* B, -1)
            #print(inputs.shape)
            #assert False, "analyzing shape first"
            inputs = inputs.to(device)
            
            # Generate latent representations
            latents = model.encoder(inputs)
            latents = latents.reshape(A,B,-1)
            latent_vectors.append(latents.cpu())  # Move to CPU to save GPU memory
            
    # Concatenate all batches into single tensor
    return torch.cat(latent_vectors, dim=0)
    #return latents

# Usage example
if __name__ == "__main__":
    # 1. Initialize model (make sure dimensions match training)
    input_dim = 21
    hidden_dim = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sensor_groups = create_sensor_groups()
    sensor_groups = {"s1_g1": sensor_groups["s1_g1"], "s1_g2": sensor_groups["s1_g2"], "s1_g3": sensor_groups["s1_g3"]}
    partition:str="A"
    
    for sequence_size in [6,5,4]:
        for op_prof in [1, 2, 3]:
            for sensor_group, sensors in sensor_groups.items():
                #print(sensor_group)
                #print(sensors)
                dataset = load_dataset(sequence_size, op_prof, sensor_group, interpolation=14)
                model = OvercompleteAutoencoder(len(sensors), 50)
                # load the model
                model_path = filepath = "models/model_weights/" + "Auto" + "/" + partition + "/" + sensor_group + "/" + str(sequence_size) + "/set_" + str(1) + "_op_prof_" + str(op_prof) + "_seq_" + str(sequence_size) + ".pt"
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
                # Move model to device
                model.to(device)
                
                # Dataloader
                # Split into train and validation
                # train the autoencoder on everything but the test data all at once
                #print(hyperparameters)
                loader = DataLoader(dataset, batch_size=40, shuffle=False) # shuffle is FALSE because we want the same ordering

                # make new dataset
                latent_data = generate_latent_representations(model, loader, device)
                #print(f"Generated latent vectors shape: {latent_data.shape}")
                #assert False, "analyzing shape"
                filepath = "data/processed_data/Autoencoder/14->50/" + partition + "/" + sensor_group + "/op_prof_" + str(op_prof) + "/dataset_seq_" + str(sequence_size) + ".pt"
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                torch.save(latent_data, filepath)
                
                #write to new data
                
    # 2. Load your trained weights
    # model.load_state_dict(torch.load('path/to/trained_model.pth'))
    
    # 3. Get your dataloader (replace with actual dataloader)
    # train_loader = DataLoader(...) 
    
    # # 4. Generate latent representations
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # latent_data = generate_latent_representations(model, train_loader, device)
    
    # print(f"Generated latent vectors shape: {latent_data.shape}")