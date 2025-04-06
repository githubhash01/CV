from torch import nn
"""
Defining Autoencoder Model
"""
class Autoencoder(nn.Module):

    name = "autoencoder"
    type = "autoencoder"

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # [B,16,H,W]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B,16,H/2,W/2]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [B,32,H/2,W/2]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B,32,H/4,W/4]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B,64,H/4,W/4]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B,64,H/8,W/8]
        )
        # Decoder: Mirror the encoder.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # [B,32,H/4,W/4]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # [B,16,H/2,W/2]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),  # [B,3,H,W]
            nn.Sigmoid()  # Output in [0,1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

