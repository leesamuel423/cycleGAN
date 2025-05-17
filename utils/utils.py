"""
Utility functions for CycleGAN training, focusing on weight initialization and learning rate scheduling.
"""

from torch.optim import lr_scheduler
import torch.nn as nn


def weights_init_normal(m):
    """
    initialize network weights with normal distribution.

    Args:
        m: PyTorch module (layer) to initialize

    different initialization for different layer types:
    - conv layers: mean=0.0, std=0.02
    - instanceNorm layers: mean=1.0, std=0.02 for weight, 0 for bias
    """
    # check if this is a Conv2d or ConvTranspose2d layer
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # initialize weights with normal distribution
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # initialize bias to zero if it exists
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    # check if this is an InstanceNorm2d layer
    elif isinstance(m, nn.InstanceNorm2d):
        # weight initialized around 1.0 (for scaling)
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        # bias initialized to 0 (no shifting)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def get_scheduler(optimizer, n_epochs, n_epochs_decay):
    """
    create learning rate scheduler with linear decay.

    Args:
        optimizer: PyTorch optimizer to schedule
        n_epochs: # of epochs with constant lr
        n_epochs_decay: # of epochs to linearly decay lr to zero

    Returns:
        scheduler: learning rate scheduler

    schedule:
    - first n_epochs: lr stays constant
    - next n_epochs_decay: lr decays linearly to 0
    - total training: n_epochs + n_epochs_decay epochs
    """

    def lambda_rule(epoch):
        """
        calculate lr multiplier for current epoch.

        returns 1.0 for first n_epochs (constant lr),
        then linearly decreases to 0 over n_epochs_decay epochs.
        """
        # calculate linear decay factor
        lr_l = 1.0 - max(0, epoch + 1 - n_epochs) / float(n_epochs_decay + 1)
        return lr_l

    # create scheduler with our custom rule
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


"""
=== Module Documentation ===
weights_init_normal(m): Initialize network weights using a specific normal distribution
    - Proper initialization prevents vanishing/exploding gradients
    - Normal distribution with std=0.02 is empirically proven for GANs
    - Different layers need different initialization strategies
   
    layer-specific strategies:
    a) Conv2d/ConvTranspose2d:
      - Weights: Normal(0.0, 0.02) - small variance prevents saturation
      - Bias: Constant(0.0) - start with no bias
   
    b) InstanceNorm2d:
      - Weights: Normal(1.0, 0.02) - centered at 1.0 for scaling
      - Bias: Constant(0.0) - start with no shift
   
    - std=0.02: found empirically to work well for GAN training
    - prevents both dead neurons and gradient explosion
    - helps networks converge faster and more stably

get_scheduler(optimizer, n_epochs, n_epochs_decay): create a learning rate scheduler for training
   scheduling strategy:
    - first n_epochs: keep learning rate constant (initial training)
    - next n_epochs_decay: linearly decay to zero (fine-tuning)
    - total training: n_epochs + n_epochs_decay epochs
   

   - initial constant LR allow model to learn main features
   - linear decay, gradually refine details without overshooting
   
   example timeline (n_epochs=100, n_epochs_decay=100):
   - Epochs 1-100: LR = initial_lr
   - Epochs 101-200: LR linearly decays from initial_lr to 0
   - Helps prevent overfitting in later epochs

"""

