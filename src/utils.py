from typing import Tuple, List
from typing import Sequence
import os
import numpy as np
from PIL import Image, ImageDraw
import pickle
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# get the parent directory of the current file
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(os.path.dirname(FILE_PATH), "Dataset")
CACHE_PATH = os.path.join(FILE_PATH, "cached_dataset")
TEST_PATH = os.path.join(DATASET_PATH, "Test")
TRAIN_PATH = os.path.join(DATASET_PATH, "TrainVal")
WEIGHTS_PATH = os.path.join(FILE_PATH, "weights")
INFERENCE_PATH = os.path.join(FILE_PATH, "inference_results")

AUG_DATASET_PATH = os.path.join(os.path.dirname(FILE_PATH), "augmented_data")
AUG_CACHE_PATH = os.path.join(FILE_PATH, "augmented_cached_dataset")

"""
Loading and caching datasets from the original dataset directory
"""
# Builds a dataset from the specified path
def build_dataset(dataset_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    dataset = []

    image_path = os.path.join(dataset_path, "color")
    mask_path = os.path.join(dataset_path, "label")

    for filename in os.listdir(image_path):
        # find the mask with the same name
        mask_filename = filename.replace("color", "label")
        mask_filename = mask_filename.replace(".jpg", ".png")

        # load the image and mask as PIL images
        image = Image.open(os.path.join(image_path, filename)).convert('RGB')
        mask = Image.open(os.path.join(mask_path, mask_filename)).convert('RGB')

        # convert the image and mask to numpy arrays and append to the dataset
        dataset.append((np.array(image), np.array(mask)))

    return dataset

# Caches training and testing datasets in the cache directory
def cache_datasets() -> None:
    # Ensure the cache directory exists
    os.makedirs(CACHE_PATH, exist_ok=True)

    # Caches a dataset in the cache directory
    def cache_dataset(dataset, filename):
        filepath = os.path.join(CACHE_PATH, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset cached to {filepath}")

    # Build the datasets for training and testing
    training_set = build_dataset(TRAIN_PATH)
    aug_training_set = build_dataset(AUG_DATASET_PATH)
    testing_set = build_dataset(TEST_PATH)

    # Cache the datasets
    cache_dataset(training_set, "training_set.pkl")
    cache_dataset(aug_training_set, "aug_training_set.pkl")
    cache_dataset(testing_set, "test_set.pkl")

# Loads the dataset from a .npy file in the cache directory
def get_cached_dataset(filename) -> List[Tuple[np.ndarray, np.ndarray]]:
    filepath = os.path.join(CACHE_PATH, filename)
    dataset = np.load(filepath, allow_pickle=True)
    # Convert the numpy object array back to a Python list
    if isinstance(dataset, np.ndarray):
        dataset = dataset.tolist()
    print(f"Dataset loaded from {filepath}")
    return dataset

"""
Defining Custom Torch Dataset and DataLoader
"""
class SegmentationDataset(Dataset):
    def __init__(self, data):
        """
        data: list of tuples (image, mask) where image and mask are NumPy arrays.
        """
        self.data = data

        # Define the image transform.
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # Define the mask transform.
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask = self.data[idx]

        # Convert NumPy arrays to PIL Images
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        # Apply transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        return image, mask

# returns a custom dataset built from training data
def get_training_dataloader():
    """
    Returns a DataLoader for the training dataset.
    """

    # returns a custom dataset for training
    def load_training_set() -> List[Tuple[np.ndarray, np.ndarray]]:
        # returns a custom dataset
        dataset = get_cached_dataset("training_set.pkl")
        return dataset

    training_set = load_training_set()
    custom_dataset = SegmentationDataset(training_set[0:100])
    dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=8, shuffle=True)
    return dataloader

# returns a custom dataset built from testing data
def get_testing_dataloader():
    """
    Returns a DataLoader for the testing dataset.
    """

    # returns a custom dataset for testing
    def load_test_set() -> List[Tuple[np.ndarray, np.ndarray]]:
        # returns a custom dataset
        dataset = get_cached_dataset("test_set.pkl")
        return dataset

    testing_set = load_test_set()
    #testing_set = testing_set[0:100]
    custom_dataset = SegmentationDataset(testing_set)
    dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=8, shuffle=False)
    return dataloader

"""
Saving and loading model weights
"""
# save weights into the weights directory
def save_weights(model) -> None:
    weights_path = os.path.join(WEIGHTS_PATH, model.name + ".pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

# load weights onto a given model
def load_weights(model: torch.nn.Module) -> torch.nn.Module:
    weights_path = os.path.join(WEIGHTS_PATH, model.name + ".pth")
    # load the weights for CPU
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    return model

"""
Saving and loading inference results
"""
# save inference results into the inference directory [predictions and targets]
def save_inference_results(model: torch.nn.Module, inference_results: Tuple[torch.Tensor, torch.Tensor]) -> None:
    # Create a filename using the model's name.
    filepath = os.path.join(INFERENCE_PATH, f"{model.name}_inference.pkl")

    # Save predictions using pickle.
    with open(filepath, "wb") as f:
        pickle.dump(inference_results, f)

    print(f"Inference results saved to {filepath}")

# load inference results from the inference directory
def load_inference_results(model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create a filename using the model's name.
    filepath = os.path.join(INFERENCE_PATH, f"{model.name}_inference.pkl")

    # Load predictions using pickle.
    with open(filepath, "rb") as f:
        inference_results = pickle.load(f)

    print(f"Inference results loaded from {filepath}")
    return inference_results

"""
Displaying images and masks
"""

def display_img(img: torch.Tensor | np.ndarray) -> None:
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(img)  # assuming img_tensor has shape (3, 256, 256)
    img_pil.show()  # This opens the image using the default image viewer.

# displays two images side by side
def display_img_img(img_a: torch.Tensor | np.ndarray, img_b: torch.Tensor | np.ndarray) -> None:
    # Convert images to PIL Images.
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(img_a)
    rec_img_pil = to_pil(img_b)

    boundary_color: tuple = (255, 255, 255)
    boundary_width: int = 3

    # Determine the size for the combined image.
    total_width = img_pil.width + rec_img_pil.width + boundary_width
    max_height = max(img_pil.height, rec_img_pil.height)

    # Create a new image with a white background.
    combined_img = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

    # Paste the first image at (0,0).
    combined_img.paste(img_pil, (0, 0))

    # Draw the boundary.
    draw = ImageDraw.Draw(combined_img)
    boundary_box = (img_pil.width, 0, img_pil.width + boundary_width - 1, max_height)
    draw.rectangle(boundary_box, fill=boundary_color)

    # Paste the second image after the boundary.
    combined_img.paste(rec_img_pil, (img_pil.width + boundary_width, 0))

    # Display the combined image.
    combined_img.show()

# displays a predicted mask and its corresponding ground truth mask
def display_mask_predicted(predicted: torch.Tensor | np.ndarray, mask_gt: torch.Tensor | np.ndarray):
    """
    Displays an image and its corresponding mask side by side.
    """

    def map_labels_to_color(label_tensor: torch.Tensor) -> np.ndarray:
        # Define your colormap: background=0 (black), cat=1 (red), dog=2 (green)
        colormap = {
            0: (0, 0, 0),  # background (black)
            1: (128, 0, 0),  # cat (red)
            2: (0, 128, 0),  # dog (green)
        }
        # Convert label tensor to numpy array.
        labels_np = label_tensor.cpu().numpy()
        H, W = labels_np.shape
        rgb_img = np.zeros((H, W, 3), dtype=np.uint8)
        for cls, color in colormap.items():
            rgb_img[labels_np == cls] = color

        return rgb_img

    def remove_white_boundary(mask: torch.Tensor) -> torch.Tensor:
        """
        Given a mask tensor of shape [3, H, W] with values in [0, 1],
        replace any white pixel ([1, 1, 1]) with black ([0, 0, 0]).
        """
        # Create a boolean mask for white pixels.
        white_pixels = torch.all(mask == 1.0, dim=0)  # shape: [H, W]
        # Set all channels of white pixels to 0 (black)
        mask[:, white_pixels] = 0.0

        return mask


    predicted_mask = torch.argmax(predicted, dim=0)
    predicted_mask = map_labels_to_color(predicted_mask)

    # get rid of the white for the ground truth mask
    mask = remove_white_boundary(mask_gt)

    # now display the two images
    display_img_img(mask, predicted_mask)

def main():
    cache_datasets()

if __name__ == "__main__":
    main()
