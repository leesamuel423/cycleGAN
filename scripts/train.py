"""Training script for CycleGAN on horse2zebra dataset."""

import itertools
import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

# add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ResnetGenerator, PatchGANDiscriminator
from utils.dataset import ImageFolderDataset
from utils.image_pool import ImagePool
from utils.utils import get_scheduler, weights_init_normal


def main():
    # set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data paths and preprocessing
    data_root = os.path.join(os.path.dirname(__file__), "..", "dataset")
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # resize all images to 256x256
            transforms.ToTensor(),  # convert PIL image to tensor
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # normalize to [-1, 1]
        ]
    )

    # create datasets for both domains
    dataset_A = ImageFolderDataset(
        data_root, transform=transform, mode="train", domain="A"
    )
    dataset_B = ImageFolderDataset(
        data_root, transform=transform, mode="train", domain="B"
    )

    # create data loaders (batch_size=1 is standard for CycleGAN)
    dataloader_A = DataLoader(dataset_A, batch_size=1, shuffle=True)
    dataloader_B = DataLoader(dataset_B, batch_size=1, shuffle=True)

    # initialize networks
    netG_A2B = ResnetGenerator().to(device)  # generator: A -> B
    netG_B2A = ResnetGenerator().to(device)  # generator: B -> A
    netD_A = PatchGANDiscriminator().to(device)  # discriminator for domain A
    netD_B = PatchGANDiscriminator().to(device)  # discriminator for domain B

    # apply weight initialization
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # define loss functions
    criterion_GAN = nn.MSELoss()  # least squares GAN loss
    criterion_cycle = nn.L1Loss()  # L1 loss for cycle consistency
    criterion_identity = nn.L1Loss()  # L1 loss for identity mapping

    # setup optimizers (Adam with beta1=0.5 is standard for GANs)
    optimizer_G = torch.optim.Adam(
        itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
        lr=0.0002,
        betas=(0.5, 0.999),
    )
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # learning rate schedulers (constant for first 100 epochs, then linear decay)
    lr_scheduler_G = get_scheduler(optimizer_G, n_epochs=100, n_epochs_decay=100)
    lr_scheduler_D_A = get_scheduler(optimizer_D_A, n_epochs=100, n_epochs_decay=100)
    lr_scheduler_D_B = get_scheduler(optimizer_D_B, n_epochs=100, n_epochs_decay=100)

    # init image pools for training stability
    fake_A_pool = ImagePool()
    fake_B_pool = ImagePool()

    # labels for real and fake images (not used but kept for clarity)
    real_label = 1.0
    fake_label = 0.0

    # training loop
    for epoch in range(200):
        # iterate through both datasets simultaneously
        for i, (data_A, data_B) in enumerate(zip(dataloader_A, dataloader_B)):
            # move data to device
            real_A = data_A.to(device)
            real_B = data_B.to(device)

            # --- forward pass ---

            # generate fake images
            fake_B = netG_A2B(real_A)  # A -> B
            rec_A = netG_B2A(fake_B)  # B -> A (reconstruction)
            fake_A = netG_B2A(real_B)  # B -> A
            rec_B = netG_A2B(fake_A)  # A -> B (reconstruction)

            ### generator losses ###

            # identity loss: G_A2B(B) should look like B
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0
            # identity loss: G_B2A(A) should look like A
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0

            # GAN loss: D_B(G_A2B(A)) should be 1 (fool discriminator)
            pred_fake_B = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))
            # GAN loss: D_A(G_B2A(B)) should be 1 (fool discriminator)
            pred_fake_A = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

            # cycle consistency loss: A -> B -> A should be A
            loss_cycle_A = criterion_cycle(rec_A, real_A) * 10.0
            # cycle consistency loss: B -> A -> B should be B
            loss_cycle_B = criterion_cycle(rec_B, real_B) * 10.0

            # total generator loss
            loss_G = (
                loss_identity_A
                + loss_identity_B
                + loss_GAN_A2B
                + loss_GAN_B2A
                + loss_cycle_A
                + loss_cycle_B
            )

            # update generators
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # --- discriminator A losses ---

            # real loss: D_A(A) should be 1
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

            # fake loss: D_A(G_B2A(B)) should be 0
            fake_A_detached = fake_A_pool.query(fake_A.detach())
            pred_fake = netD_A(fake_A_detached)
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

            # average discriminator loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            # update discriminator A
            optimizer_D_A.zero_grad()
            loss_D_A.backward()
            optimizer_D_A.step()

            # --- discriminator B losses ---

            # real loss: D_B(B) should be 1
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

            # fake loss: D_B(G_A2B(A)) should be 0
            fake_B_detached = fake_B_pool.query(fake_B.detach())
            pred_fake = netD_B(fake_B_detached)
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

            # average discriminator loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5

            # update discriminator B
            optimizer_D_B.zero_grad()
            loss_D_B.backward()
            optimizer_D_B.step()

            # print progress every 100 iterations
            if i % 100 == 0:
                print(
                    f"Epoch {epoch} Iter {i} Loss_G {loss_G.item():.4f} "
                    f"Loss_D_A {loss_D_A.item():.4f} Loss_D_B {loss_D_B.item():.4f}"
                )

        # update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # save checkpoints every epoch
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(netG_A2B.state_dict(), f"checkpoints/netG_A2B_epoch_{epoch+1}.pth")
        torch.save(netG_B2A.state_dict(), f"checkpoints/netG_B2A_epoch_{epoch+1}.pth")
        torch.save(netD_A.state_dict(), f"checkpoints/netD_A_epoch_{epoch+1}.pth")
        torch.save(netD_B.state_dict(), f"checkpoints/netD_B_epoch_{epoch+1}.pth")
        torch.save(
            optimizer_G.state_dict(), f"checkpoints/optimizer_G_epoch_{epoch+1}.pth"
        )
        torch.save(
            optimizer_D_A.state_dict(), f"checkpoints/optimizer_D_A_epoch_{epoch+1}.pth"
        )
        torch.save(
            optimizer_D_B.state_dict(), f"checkpoints/optimizer_D_B_epoch_{epoch+1}.pth"
        )
        torch.save({"epoch": epoch + 1}, f"checkpoints/epoch_state_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    main()

"""
data loading:
- loads unpaired images from domain A (e.g., horses) and domain B (e.g., zebras)
- applies standard preprocessing: resize to 256×256, normalize to [-1, 1]
- batch size of 1 is typical for GANs to maintain training stability

network architecture:
- two generators: G_A2B (A→B) and G_B2A (B→A)
- two discriminators: D_A (judges domain A) and D_B (judges domain B)

cycleGAN uses multiple loss components:
    a) GAN Loss (adversarial):
        - makes generated images look realistic to discriminators
        - uses MSE loss (least squares GAN) for stability
        - applied to both translation directions
   
    b) cycle consistency loss:
        - key innovation of CycleGAN
        - ensures A→B→A ≈ A and B→A→B ≈ B
        - weight of 10.0 emphasizes this crucial constraint
        - prevents mode collapse and maintains content

    c) identity loss:
        - ensures G_A2B(B) ≈ B and G_B2A(A) ≈ A
        - helps preserve color/tones of input domain
        - weight of 5.0 provides moderate regularization


training strat:
    generator training:
    - forward pass through both directions
    - compute all three loss types
    - single optimizer for both generators (efficient)

    discriminator training:
    - separate training for each discriminator
    - uses image pool for training stability
    - standard real/fake classification task

optimizations:
- Adam optimizer with lr=0.0002, beta1=0.5 (standard for GANs)
- linear learning rate decay after 100 epochs
- total 200 epochs (100 constant + 100 decay)

checkpointing:
- saves all models and optimizers every epoch
- allows resuming training and using intermediate results

- cycle consistency enforces bijective mapping without paired data
- adversarial training ensures realistic outputs
- identity loss prevents unnecessary changes
- balanced training keeps both directions improving together
"""

