"""
Inference script for CycleGAN, used to generate translated images from a trained model.

Loads a trained generator model and translates all images from the test set of one domain to another.
By default, it translates from domain A to domain B (e.g., horses to zebras).

To translate from B to A instead:
1. Change input_path to "testB"
2. Load "netG_B2A_epoch_200.pth" checkpoint
3. Adjust output directory name accordingly

To use a different epoch checkpoint:
- Change the checkpoint path to desired epoch
- Earlier epochs may show training progression

To process custom images:
1. Place images in a directory
2. Update input_path to that directory
3. Ensure images are in supported format (jpg, png)
"""

import os
import sys
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ResnetGenerator


def load_image(path):
    """
    load and preprocess a single image.

    Args:
        path: path to the image file

    Returns:
        preprocessed image tensor with batch dimension
    """
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # resize to match training
            transforms.ToTensor(),  # convert to tensor
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # normalize to [-1, 1]
        ]
    )
    image = Image.open(path).convert("RGB")  # load and ensure RGB format
    return transform(image).unsqueeze(0)  # add batch dimension


def main():
    # set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set input and output paths
    input_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "testA")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)  # create output directory if needed

    # load the generator model
    netG = ResnetGenerator()
    checkpoint = "checkpoints/netG_A2B_epoch_10.pth"  # using epoch 10 checkpoint
    netG.load_state_dict(torch.load(checkpoint, map_location=device))
    netG.to(device)
    netG.eval()  # set to evaluation mode (disables dropout, etc.)

    # process all images in the test directory
    for img_name in os.listdir(input_path):
        # load and preprocess image
        img = load_image(os.path.join(input_path, img_name)).to(device)

        # generate translation without computing gradients
        with torch.no_grad():
            fake = netG(img)

        # denormalize from [-1, 1] to [0, 1] for saving
        fake = (fake + 1) / 2.0

        # save the generated image
        save_image(fake, os.path.join(output_dir, img_name))
        print(f"Processed {img_name}")


if __name__ == "__main__":
    main()

"""
image loading and preprocessing:
- load_image() function handles individual image loading
- same preprocessing as training: resize to 256×256, normalize to [-1, 1]
- unsqueeze(0) adds batch dimension for model compatibility

model setup:
- loads only the generator (discriminators not needed for inference)
- uses checkpoint from epoch 200 (final model after full training)
- sets model to eval mode to disable dropout/batch norm training behavior
- automatically uses GPU if available

inference process:
- processes all images in testA directory
- torch.no_grad() context prevents gradient computation (saves memory)
- generator outputs are in [-1, 1] range
- denormalization: (fake + 1) / 2.0 converts back to [0, 1] range

output handling:
- creates 'outputs' directory for results
- preserves original filenames for easy comparison
- uses torchvision's save_image for proper image formatting


- normalize to [-1, 1] b/c centered data helps training stability & Tanh activation in generator matches this range

- eval mode important b/c disables dropout layers (if any). Uses running statistics for batch/instance norm and ensures deterministic, reproducible results

Memory optimization:
- no_grad() prevents storing computation graph
- processes one image at a time
- suitable for large test sets

"""

