from torch import optim
from torch import nn
import torch
from tqdm import tqdm
from utils import  save_weights, load_weights, get_training_dataloader, get_validation_dataloader

"""
Generic training function
"""
def train_model(model, criterion, optimizer, num_epochs=10):
    training_dataloader = get_training_dataloader()
    validation_dataloader = get_validation_dataloader()

    minimum_loss = float('inf')

    print(f"Training {model.name}...")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()  # set model to training mode once per epoch
        for inputs, targets in training_dataloader:
            # For autoencoders, targets are the same as inputs
            if model.type == "autoencoder":
                targets = inputs

            if targets.dim() == 4 and targets.size(1) == 1:
                targets = targets.squeeze(1).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # --- Validation Phase ---
        model.eval()  # set model to evaluation mode once per epoch
        val_loss_total = 0.0
        num_batches = 0

        # Disable gradient computation during validation
        with torch.no_grad():
            for inputs, targets in tqdm(validation_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                # For autoencoders, targets are the same as inputs
                if model.type == "autoencoder":
                    targets = inputs

                if targets.dim() == 4 and targets.size(1) == 1:
                    targets = targets.squeeze(1).long()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss_total += loss.item()
                num_batches += 1

        avg_val_loss = val_loss_total / num_batches

        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")

        # Save the model if validation loss improved
        if avg_val_loss < minimum_loss:
            minimum_loss = avg_val_loss
            save_weights(model)


"""
Specific functions for Autoencoders
"""
def train_autoencoder():
    from models.autoencoder import Autoencoder
    autoencoder_model = Autoencoder()
    train_model(autoencoder_model, nn.MSELoss(), optimizer=optim.Adam(autoencoder_model.parameters(), lr=1e-3))

def train_clip_autoencoder():
    pass

""" 
Specific functions for Segmentation
"""

# 1) Segmentation Model: UNet
def train_unet():
    from models.UNet import UNet
    unet_model = UNet(num_classes=3)
    train_model(unet_model, nn.CrossEntropyLoss(ignore_index=255), optimizer=optim.Adam(unet_model.parameters(), lr=1e-3))

# 2) Segmentation Model: Autoencoder
def train_autoencoder_segmentation():
    from models.autoencoder import Autoencoder
    from models.AutoencoderSegment import SegmentationModel

    # load in pre-trained autoencoder
    autoencoder_model = Autoencoder()
    autoencoder_model = load_weights(autoencoder_model)
    segmentation_model = SegmentationModel(encoder=autoencoder_model.encoder, num_classes=3)

    train_model(segmentation_model, nn.CrossEntropyLoss(ignore_index=255), optim.Adam(segmentation_model.parameters(), lr=1e-3))

# 3) Segmentation Model: CLIP Autoencoder
def train_clip_autoencoder_segmentation():
    pass

if __name__ == "__main__":
    train_autoencoder()
    #train_unet()
    #train_autoencoder_segmentation()