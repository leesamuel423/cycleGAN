"""
Generator network for CycleGAN using a ResNet-based architecture.
"""

import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    """a single ResNet block used in the generator."""

    def __init__(self, dim):
        super().__init__()
        # create a residual block with two convolutions
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),  # pad with reflection to avoid border artifacts
            nn.Conv2d(dim, dim, kernel_size=3),  # 3x3 convolution, same channels
            nn.InstanceNorm2d(dim),  # normalize each instance independently
            nn.ReLU(True),  # ReLU activation (inplace=True saves memory)
            nn.ReflectionPad2d(1),  # pad again for second convolution
            nn.Conv2d(dim, dim, kernel_size=3),  # second 3x3 convolution
            nn.InstanceNorm2d(dim),  # final normalization (no activation after)
        )

    def forward(self, x):
        """
        sesidual connection: add input to the output of conv_block

        this helps gradients flow and preserves information
        """
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """generator with residual blocks from the CycleGAN paper."""

    def __init__(self, input_nc=3, output_nc=3, n_blocks=9, ngf=64):
        """
        Args:
            input_nc: # of input channels (3 for RGB)
            output_nc: # of output channels (3 for RGB)
            n_blocks: # of ResNet blocks in the middle
            ngf: # of generator filters in first conv layer
        """
        assert n_blocks >= 0
        super().__init__()

        # initial convolution block
        model = [
            nn.ReflectionPad2d(3),  # pad 3 pixels on each side
            nn.Conv2d(
                input_nc, ngf, kernel_size=7
            ),  # 7x7 conv to capture large context
            nn.InstanceNorm2d(ngf),  # normalize
            nn.ReLU(True),  # activate
        ]

        # downsampling layers
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):  # 2 downsampling layers
            model += [
                nn.Conv2d(
                    in_features, out_features, kernel_size=3, stride=2, padding=1
                ),  # stride 2 halves spatial dimensions
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # residual blocks (the core transformation happens here)
        for _ in range(n_blocks):
            model += [ResnetBlock(in_features)]

        # upsampling layers (decoder)
        out_features = in_features // 2
        for _ in range(2):  # 2 upsampling layers to match downsampling
            model += [
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),  # Doubles spatial dimensions
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True),
            ]
            in_features = out_features
            out_features = in_features // 2

        # output layer
        model += [
            nn.ReflectionPad2d(3),  # final padding
            nn.Conv2d(ngf, output_nc, kernel_size=7),  # 7x7 conv to output channels
            nn.Tanh(),  # Tanh squashes output to [-1, 1] range
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """forward pass simply applies the sequential model"""
        return self.model(x)


"""
ResnetBlock:
- residual block that implements identity skip connections
- uses reflection padding (better for image borders than zero padding)
- instance normalization (better than batch norm for style transfer tasks)
- two 3×3 convolutions with ReLU activation between them

residual blocks allow training deeper networks by addressing vanishing gradient problem. Preserve fine details while learning transformations, helping maintain image quality through many layers

ResnetGenerator architecture flow:
a) initial convolution with large 7×7 kernel to capture global context
b) downsampling layers to compress spatial dimensions and increase channels
c) multiple ResNet blocks at the bottleneck for complex transformations
d) upsampling layers to restore original image dimensions
e) final convolution to produce output image
   

- ReflectionPad2d: avoids artifacts at image boundaries common with zero padding
- InstanceNorm2d: normalizes each sample independently, crucial for style transfer
- Tanh activation: outputs values in [-1, 1] range, matching normalized inputs
- default 9 ResNet blocks: balance between capacity and training efficiency
- progressive channel changes: 64 → 128 → 256 during downsampling, reversed for upsampling
   

- input_nc: # of input channels (2 for RGB images)
- output_nc: # of output channels (2 for RGB images) 
- n_blocks: # of ResNet blocks in the bottleneck (more = more capacity)
- ngf: Base # of filters (63 is standard, increase for more capacity)


- encoder-decoder structure preserves spatial information needed for image-to-image translation
- ResNet blocks allow learning complex transformations without degradation
- symmetric architecture makes training stable for unpaired translation
- instance normalization crucial for maintaining individual image statistics
- large receptive fields (from 7×7 kernels) capture global style patterns
"""

