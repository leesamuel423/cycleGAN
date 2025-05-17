"""
This file implements the dataset loader for CycleGAN training and testing.

This is based off the expected directory structure:
- dataset/
    - trainA/    (training images from domain A)
    - trainB/    (training images from domain B)
    - testA/     (test images from domain A)
    - testB/     (test images from domain B)
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageFolderDataset(Dataset):
    """dataset reading images from trainA/trainB/testA/testB directories."""

    def __init__(self, root, transform=None, mode="train", domain="A"):
        """
        initialize the dataset.

        Args:
            root: root directory containing subdirectories
            transform: optional transforms to apply to images
            mode: either "train" or "test"
            domain: either "A" or "B" for the two domains
        """
        assert mode in {"train", "test"}
        self.transform = transform

        # construct subdirectory name based on mode and domain
        subdir = "train" + domain if mode == "train" else "test" + domain
        self.dir = os.path.join(root, subdir)

        # get sorted list of image filenames for reproducibility
        self.images = sorted(os.listdir(self.dir))

    def __len__(self):
        """return the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        get a single image by index.

        Args:
            idx: index of the image to retrieve

        Returns:
            image: transformed image tensor
        """
        # construct full path to image
        img_path = os.path.join(self.dir, self.images[idx])

        # open image and convert to RGB(for greyscale)
        image = Image.open(img_path).convert("RGB")

        # apply transforms <optional> (resize, normalize, etc.)
        if self.transform is not None:
            image = self.transform(image)

        return image


"""
Separate A/B domains
    - Images in A and B don't need to correspond (unpaired)
    - Model learns to map between distributions of A and B

Implementation details:
    - Inherits from PyTorch Dataset for compatibility with DataLoader
    - Sort filenames for reproducibility across runs
    - Always convert to RGB to handle grayscale images consistently
    - Delegate preprocessing to transforms for flexibility

Rationale:
   - No need for paired data or complex alignment
   - Each domain just needs to represent its distribution well
   - Random sampling during training provides variety
   - File sorting ensures consistent ordering for evaluation

"""
