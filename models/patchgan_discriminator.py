"""
This file implements PatchGAN discriminator
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator"""

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """
        Args:
            input_nc: # of input channels (3 for RGB)
            ndf: # of discriminator filters in first conv layer
            n_layers: # of conv layers (3 creates ~70x70 receptive field)
        """
        super().__init__()

        # build layers list starting with first layer (no normalization)
        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),  # LeakyReLU with negative slope 0.2
        ]

        # track filter multipliers for progressive channel increase
        nf_mult = 1
        nf_mult_prev = 1

        # add intermediate layers with normalization
        for n in range(1, n_layers):  # n_layers-1 intermediate layers
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)  # cap channel multiplier at 8
            layers += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),  # downsample by 2
                nn.InstanceNorm2d(ndf * nf_mult),  # normalize per instance
                nn.LeakyReLU(0.2, True),
            ]

        # second to last layer (stride 1, no downsampling)
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            nn.Conv2d(
                ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1
            ),  # stride 1, maintains size
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        # output layer (single channel, no activation)
        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        # stack all layers into sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        forward pass outputs a grid of real/fake predictions.

        each spatial location corresponds to a patch classification.
        """
        return self.model(x)


"""
PatchGAN:
- instead of classifying entire images as real/fake, classifies overlapping patches
- each output pixel represents whether a 70×70 patch is real or fake

Network Structure:
1. initial layer: no normalization, just Conv + LeakyReLU
2. middle layers: Conv + InstanceNorm + LeakyReLU with stride 2 (downsampling)
3. penultimate layer: Conv + InstanceNorm + LeakyReLU with stride 1
4. final layer: Conv to single channel output (no activation)

PatchGAN instead of regular discriminator bc:
- computational efficiency: fewer parameters than full-image discriminator
- better texture modeling: focuses on local image statistics
- implicit ensemble: each patch vote contributes to final loss
- translation invariance: same patch statistics expected across image

70×70 patches are large enough to capture meaningful texture patterns, but small enough to be computationally efficient

- 4×4 kernels with stride 2: standard for discriminators, provides overlapping receptive fields
- LeakyReLU(0.2): allows gradients for negative values, preventing dead neurons
- no normalization in first layer: preserves low-level image statistics
- InstanceNorm in other layers: helps training stability
- channel progression (64 → 128 → 256 → 512): gradually increases capacity
- no sigmoid activation: modern GANs use least-squares or Wasserstein loss

- input_nc: # of input channels (3 for RGB)
- ndf: Base discriminator filters (64 is standard)
- n_layers: # of conv layers (3 gives 70×70 receptive field)

receptive field calculation:
- each 4×4 conv with stride 2 roughly doubles the receptive field
- with 3 layers: roughly 70×70 pixel patches are evaluated, so "70x70 PatchGAN"

- Local texture consistency is key for style transfer
- Patch-based approach handles varying image sizes
- Multiple patch evaluations provide stable training signal
- Computational efficiency allows for larger generators
"""

