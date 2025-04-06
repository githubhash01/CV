from torch import nn
import torch

class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Example architecture: sequence of ConvTranspose2d layers to upsample 32x
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 384, kernel_size=2, stride=2),  # 7x7 -> 14x14
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),          # 14 -> 28
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),           # 28 -> 56
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),            # 56 -> 112
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, num_classes, kernel_size=2, stride=2)    # 112 -> 224
            # (No activation here; use log-softmax or apply CrossEntropyLoss on logits)
        )
    def forward(self, feat_map):
        return self.decode(feat_map)

class CLIPSegModel(nn.Module):
    def __init__(self, clip_encoder, num_classes):
        super().__init__()
        self.clip_encoder = clip_encoder # frozen CLIP backbone
        self.decoder = SegmentationDecoder(in_channels=768, num_classes=num_classes)

    def forward(self, pixel_values):

        # Pass through CLIP encoder (no grad)
        with torch.no_grad():
            outputs = self.clip_encoder(pixel_values=pixel_values)

        patch_feats = outputs.last_hidden_state[:, 1:, :]            # drop CLS, shape [B, N_patches, C]
        B, N, C = patch_feats.shape
        H_patch = W_patch = int(N**0.5)
        feat_map = patch_feats.transpose(1, 2).reshape(B, C, H_patch, W_patch)  # [B, C, H_patch, W_patch]
        # Decoder: upsample to segmentation logits [B, num_classes, H, W]
        logits = self.decoder(feat_map)

        return logits