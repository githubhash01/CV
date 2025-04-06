from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import tqdm
from utils import  save_weights, load_weights, get_training_dataloader

"""
Generic training function
"""
def train_model(model, criterion, optimizer, num_epochs=10):

    dataloader = get_training_dataloader()

    print(f"Training {model.name}...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for inputs, targets in pbar:
            optimizer.zero_grad()
            outputs = model(inputs)

            # For autoencoder, we use the inputs as targets.
            if model.type == "autoencoder":
                targets = inputs

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            # Update the progress bar with the current loss.
            pbar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Save the model's state dictionary after training.
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
    train_model(unet_model, nn.CrossEntropyLoss(), optimizer=optim.Adam(unet_model.parameters(), lr=1e-3))

# 2) Segmentation Model: Autoencoder
def train_autoencoder_segmentation():
    from models.autoencoder import Autoencoder
    from models.AutoencoderSegment import SegmentationModel

    # load in pre-trained autoencoder
    autoencoder_model = Autoencoder()
    autoencoder_model = load_weights(autoencoder_model)
    segmentation_model = SegmentationModel(encoder=autoencoder_model.encoder, num_classes=3)

    train_model(segmentation_model, nn.CrossEntropyLoss(), optim.Adam(segmentation_model.parameters(), lr=1e-3))

# 3) Segmentation Model: CLIP Autoencoder
def train_clip_autoencoder_segmentation():
    pass

if __name__ == "__main__":
    train_autoencoder()
    #train_unet()
    #train_autoencoder_segmentation()