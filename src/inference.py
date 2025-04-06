import torch
from utils import load_weights, save_inference_results, get_testing_dataloader
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Generic inference function
"""
def run_inference(model: torch.nn.Module) -> None:
    """
    Run inference on a given model and the test dataset.

    Args:
        model: A torch.nn.Module
    Returns:
        A tensor containing the concatenated predictions from the model.
    """

    dataloader = get_testing_dataloader()
    model = load_weights(model.to(device))
    model.eval()

    all_targets = []
    predictions = []
    print(f"Running inference on {model.name}...")
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(dataloader, desc="Inference", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu())

            # For autoencoder models, we use the inputs as targets.
            if model.type == "autoencoder":
                targets = inputs
            all_targets.append(targets.cpu())

    # Concatenate all batch predictions and targets along the batch dimension.
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Save both predictions and targets as a tuple.
    inference_results = (predictions, targets)
    save_inference_results(model, inference_results)


"""
Specific functions for running inference
"""

def infer_autoencoder():
    from models.autoencoder import Autoencoder
    autoencoder_model = Autoencoder().to(device)
    run_inference(autoencoder_model)

def infer_unet():
    from models.UNet import UNet
    unet_model = UNet(num_classes=3).to(device)
    run_inference(unet_model)

def infer_autoencoder_segmentation():
    from models.autoencoder import Autoencoder
    from models.AutoencoderSegment import SegmentationModel
    autoencoder = Autoencoder()
    segmentation_model = SegmentationModel(encoder=autoencoder.encoder, num_classes=3).to(device)
    run_inference(segmentation_model)

if __name__ == "__main__":
    #infer_autoencoder()
    #infer_unet()
    infer_autoencoder_segmentation()
