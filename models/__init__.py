from .resnet_generator import ResnetGenerator
from .patchgan_discriminator import PatchGANDiscriminator

__all__ = [
    "ResnetGenerator",
    "PatchGANDiscriminator",
]

"""
This __init__.py file serves as the package initialization for the models module in the CycleGAN implementation.

ResnetGenerator: generator network that performs the actual image translation (A→B and B→A)
    - based on ResNet architecture with residual connections for better gradient flow
    - used because deep networks are needed for complex image transformations

PatchGANDiscriminator: the discriminator network that judges if images are real or fake
    - implements a PatchGAN architecture that classifies 70×70 patches
    - more efficient than full-image discriminators while maintaining quality
"""

