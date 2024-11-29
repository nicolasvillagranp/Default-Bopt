"""
Script for some utility functions.
"""


# standard libraries
import os
import shutil
import random
import tarfile
from transformers import AutoImageProcessor


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import requests
import tarfile


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Parameters
    ----------
    seed : Seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class ImagewoofDataset(Dataset):
    """
    This class is the Imagewoof Dataset.
    """

    def __init__(self, path: str) -> None:
        """
        Constructor of ImagewoofDataset.

        Args:
            path: path of the dataset.
        """
        super().__init__()
        self.path: str = path
        self.images: list[str] = os.listdir(path)
        self.labels: list[int] = [int(name.split("_")[0]) for name in self.images]
        self.processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k", use_fast=True
        )

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            length of dataset.
        """
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        This method loads an item based on the index.

        Args:
            index: index of the element in the dataset.

        Returns:
            tuple with image and label. Image dimensions:
                [channels, height, width].
        """
        img_path = os.path.join(self.path, self.images[index])
        image = Image.open(img_path).convert("RGB")  # Load image as PIL.Image
        label = self.labels[index]
        return self.processor(image, return_tensors="pt").pixel_values, label


def load_data(
    path: str, batch_size: int = 128, num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    This function returns two Dataloaders, one for train data and
    other for validation data for imagewoof dataset.

    Args:
        path: path of the dataset.
        color_space: color_space for loading the images.
        batch_size: batch size for dataloaders. Default value: 128.and
        num_workers: number of workers for loading data.
            Default value: 0.

    Returns:
        tuple of dataloaders, train, val in respective order.
    """
    # TODO
    # Create datasets
    train_dataset: Dataset = ImagewoofDataset(f"{path}/train")
    val_dataset: Dataset = ImagewoofDataset(f"{path}/val")

    # Define dataloaders
    train_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_dataloader, val_dataloader


def download_data(path: str) -> None:
    """
    This function downloads the Imagewoof dataset from GitHub and processes the images.

    Args:
        path: Path to save the data.
    """
    # Define the URL for Imagewoof dataset and the target path to save the tar file
    url: str = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz"
    target_path: str = f"{path}/imagewoof-320.tgz"

    # Download the tar file
    response = requests.get(url, stream=True)
    print(f"Downloading Imagewoof from {url}...")
    if response.status_code == 200:
        with open(target_path, "wb") as f:
            f.write(response.raw.read())
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")
        return

    # Extract the tar file
    print("Extracting data...")
    with tarfile.open(target_path, "r:gz") as tar:
        tar.extractall(path)

    # Create final save directories (train, val)
    print("Creating necessary directories...")
    os.makedirs(f"{path}/train", exist_ok=True)
    os.makedirs(f"{path}/val", exist_ok=True)

    # Define resize transformation
    transform = transforms.Resize((224, 224))

    # Loop through splits (train, val, test) and save processed data
    # loop for saving processed data
    list_splits: tuple[str, str] = ("train", "val")
    for i in range(len(list_splits)):
        list_class_dirs = os.listdir(f"{path}/imagewoof2-320/{list_splits[i]}")
        for j in range(len(list_class_dirs)):
            list_dirs = os.listdir(
                f"{path}/imagewoof2-320/{list_splits[i]}/{list_class_dirs[j]}"
            )
            for k in range(len(list_dirs)):
                image = Image.open(
                    f"{path}/imagewoof2-320/{list_splits[i]}/"
                    f"{list_class_dirs[j]}/{list_dirs[k]}"
                )
                image = transform(image)
                if image.im.bands == 3:
                    image.save(f"{path}/{list_splits[i]}/{j}_{k}.jpg")

    # Delete the tar file and extracted folder to clean up
    print("Cleaning up...")
    os.remove(target_path)
    shutil.rmtree(f"{path}/imagewoof2-320")

    print("Download and processing complete!")


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """
    This method computes accuracy from logits and labels

    Args:
        logits: batch of logits. Dimensions: [batch, number of classes]
        labels: batch of labels. Dimensions: [batch]

    Returns:
        accuracy of predictions
    """

    predictions = logits.argmax(1).type_as(labels)

    result = predictions.eq(labels).float().mean().cpu().detach()

    return result


if __name__ == "__main__":
    download_data("data")