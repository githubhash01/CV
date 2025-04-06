import torch
import clip
from PIL import Image
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import tqdm
from utils import load_training_set, save_weights, load_weights, AutoencoderDataset, SegmentationDataset, display_img
import numpy as np

TRAINING_SET = load_training_set()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Assuming AutoencoderDataset returns raw images (e.g., PIL images or tensors)
training_set = AutoencoderDataset(TRAINING_SET)
image, mask = training_set[0]

print("Original image shape:", image.shape)  # e.g. torch.Size([3, 256, 256])

# Preprocess the image: this will resize to 224x224, normalize, etc.
pil_image = ToPILImage()(image)
preprocessed_image = preprocess(pil_image).unsqueeze(0).to(device)
print("Preprocessed image shape:", preprocessed_image.shape)  # e.g. torch.Size([1, 3, 224, 224])

# preprocessed_image now has shape [1, 3, 224, 224]
class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=512, output_channels=3, image_size=224):
        super(SimpleDecoder, self).__init__()
        # Map latent vector to a feature map (e.g., 256 channels with 7x7 spatial size)
        self.fc = nn.Linear(latent_dim, 256 * 7 * 7)

        # Upsample the 7x7 feature map to image_size x image_size
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1),  # 112x112 -> 224x224
            nn.Tanh()  # Output values between -1 and 1 (assuming images are normalized accordingly)
        )

    def forward(self, x):
        # x: [B, latent_dim]
        x = self.fc(x)  # [B, 256*7*7]
        x = x.view(-1, 256, 7, 7)  # reshape to a spatial feature map
        x = self.deconv(x)  # upsample to [B, 3, 224, 224]
        return x

class ClipAutoencoder(nn.Module):
    def __init__(self):
        super(ClipAutoencoder, self).__init__()
        self.encoder = model.encode_image  # CLIP's image encoder is used as-is

    def forward(self, x):
        # encode the image and then decode it
        #latent = self.encoder(x)

        # Decode the latent representation

        return self.encoder(x)

# Create an instance of the model and pass the preprocessed image
clip_model = ClipAutoencoder().to(device)
encoded_image = clip_model(preprocessed_image)
simple_decoder = SimpleDecoder().to(device)
decoded_image = simple_decoder(encoded_image)
print("Encoded image shape:", encoded_image.shape)
print("Decoded image shape:", decoded_image.shape)


# Post process the decoded image back to 3, 256, 256 tensor from 1, 3, 224, 224

decoded_image = decoded_image.squeeze(0)  # Remove the batch dimension
decoded_image = decoded_image.permute(1, 2, 0)  # Change to HWC format
decoded_image = decoded_image.cpu().detach().numpy()  # Convert to numpy array
decoded_image = (decoded_image + 1) / 2.0  # Rescale to [0, 1]
decoded_image = decoded_image * 255  # Rescale to [0, 255]
decoded_image = decoded_image.astype('uint8')  # Convert to uint8
decoded_image = Image.fromarray(decoded_image)  # Convert to PIL image
decoded_image = decoded_image.resize((256, 256))  # Resize to original size
# convert back to tensor
decoded_image = torch.from_numpy(np.array(decoded_image)).permute(2, 0, 1)  # Convert back to tensor

display_img(decoded_image)