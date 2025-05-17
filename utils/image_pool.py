"""
Implements ImagePool (replay buffer) used in CycleGAN training to stabilize the discriminator.
"""

import random
import torch


class ImagePool:
    """history of generated images to stabilize discriminator training."""

    def __init__(self, pool_size=50):
        """
        initialize the image pool.

        Args:
            pool_size: max # of images to store (0 disables pooling)
        """
        self.pool_size = pool_size
        self.images = []  # store generated images

    def query(self, image):
        """
        return an image from pool, possibly replacing it with new one.

        Args:
            image: new generated image to potentially add to pool

        Returns:
            image to use for discriminator training (might be old or new)
        """
        # if pool size is 0, pooling is disabled
        if self.pool_size == 0:
            return image

        # if pool not full yet, just add the new image
        if len(self.images) < self.pool_size:
            self.images.append(image.detach())  # detach() prevents gradient flow
            return image

        # pool is full -> randomly decide whether to use new or old image
        if random.random() > 0.5:
            # replace a random old image with new one
            idx = random.randint(0, self.pool_size - 1)
            tmp = self.images[
                idx
            ].clone()  # get old image (clone to avoid reference issues)
            self.images[idx] = image.detach()  # store new image
            return tmp  # return old image for training

        # return the new image without storing it
        return image


"""
ImagePool addresses an important training stability issue in GANs:
    - In standard GAN training, discriminators only see latest generated images
    - Generators can exploit this by drastically changing their outputs
    - This leads to discriminator "forgetting" about past generated images -> training instability and mode collapse

replay buffer:
   - Store a pool of previously generated images
   - Sometimes show discriminator old generated images instead of just new ones
   - Prevents generator from exploiting discriminator's limited memory
   - Inspired by experience replay in reinforcement learning

Pool Management:
    - Maintains a fixed-size buffer of generated images (default 50)
    - When pool is not full, just adds new images
    - When pool is full, randomly decides to either:
         a) Return the new image (50% chance)
         b) Return old image from pool and replace it with new one (50% chance)

    - pool_size=50: large enough for diversity, small enough for memory efficiency
    - 50/50 probability: balances seeing new vs. old generated images
    - detach(): prevents gradients from flowing through stored images
    - clone(): ensures we return a copy, not a reference

So discriminator sees a mix of recent and past generated images. Generator can't suddenly change strategy to fool discriminator -> more consistent training signal for both networks. This reduces oscillations and mode collapse
- Applied separately to fake images from each generator (G_A2B and G_B2A)
- Called before passing fake images to discriminators
"""
