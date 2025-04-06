from utils import load_inference_results, display_img_img, display_mask_predicted

"""
Evaluating Autoencoder Models
"""
def evaluate_autoencoder(idx=0):
    from models.autoencoder import Autoencoder

    autoencoder_model = Autoencoder()
    inference_results = load_inference_results(autoencoder_model)

    prediction = inference_results[0][idx]
    target = inference_results[1][idx]

    display_img_img(prediction, target)

"""
Evaluating Segmentation Models
"""

def evaluate_unet(idx=8):
    from models.UNet import UNet
    unet_model = UNet(num_classes=3)

    inference_results = load_inference_results(unet_model)
    predicted_mask = inference_results[0][idx]
    mask_gt = inference_results[1][idx]

    display_mask_predicted(predicted=predicted_mask, mask_gt=mask_gt)

def evaluate_autoencoder_segmentation(idx=8):
    from models.autoencoder import Autoencoder
    from models.AutoencoderSegment import SegmentationModel
    autoencoder_model = Autoencoder()
    segmentation_model = SegmentationModel(encoder=autoencoder_model.encoder, num_classes=3)

    inference_results = load_inference_results(segmentation_model)
    predicted_mask = inference_results[0][idx]
    mask_gt = inference_results[1][idx]
    display_mask_predicted(predicted=predicted_mask, mask_gt=mask_gt)


if __name__ == "__main__":
    evaluate_unet(124)
    #evaluate_autoencoder(23)
    ##evaluate_autoencoder_segmentation(32)