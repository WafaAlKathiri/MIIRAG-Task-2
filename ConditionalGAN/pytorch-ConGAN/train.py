import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import glob
import time

from model import Generator, Discriminator, weights_init

# Hyperparameters - Experiment with these!
NUM_EPOCHS = 700
BATCH_SIZE = 8
LR = 0.0002  # Slightly higher learning rate
BETA1 = 0.5
LATENT_DIM = 100
NUM_CLASSES = 8
IMAGE_SIZE = 512
NUM_CHANNELS = 3
CHECKPOINT_INTERVAL = 50
PROGRESS_INTERVAL = 10
GRADIENT_CLIP = 1.0
RESUME_TRAINING = False
PRETRAINED_DISCRIMINATOR = True 
PRETRAINED_DISCRIMINATOR_PATH = "/kaggle/working/crack-discriminator/discriminator_epoch_250.pth"
# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path Configuration
dataset_path = "/kaggle/working/roadcrackds/Road_Crack_Dataset_Cleaned_labled/train"  #Correct data path
progress_dir = "/kaggle/working/progress"
checkpoint_dir = "/kaggle/working/model_checkpoints"

# 1. Data Loading and Preprocessing
def initialize_environment():
    os.makedirs(progress_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    return transform

def create_data_loaders(transform):
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# 2. Model Initialization
def initialize_models():
    netG = Generator(num_classes=NUM_CLASSES, latent_dim=LATENT_DIM, ngf=64, num_channels=NUM_CHANNELS).to(device)
    netD = Discriminator(num_classes=NUM_CLASSES, ndf=64, num_channels=NUM_CHANNELS).to(device)

    # Initialize weights (important for GAN stability)
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Optimizer setup
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))

    start_epoch = 0
    if RESUME_TRAINING:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            netG.load_state_dict(checkpoint['netG_state_dict'])
            netD.load_state_dict(checkpoint['netD_state_dict'])
            optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
            optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting from scratch.")

    return netG, netD, optimizerG, optimizerD, start_epoch
# 3. Training Loop
def train(netG, netD, dataloader, optimizerG, optimizerD, start_epoch):
    # Loss function - BCEWithLogitsLoss for more stable training
    criterion = nn.BCELoss()

    # Training loop
    start_time = time.time()
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time.time()

        for batch_idx, (real_images, labels) in enumerate(dataloader):
            real_images = real_images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            netD.zero_grad()
            real_outputs = netD(real_images, labels)
            real_loss = criterion(real_outputs, torch.ones_like(real_outputs))
            # Generate fake images
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_images = netG(noise, labels)
            # Classify fake images
            fake_outputs = netD(fake_images.detach(), labels)
            fake_loss = criterion(fake_outputs, torch.zeros_like(fake_outputs))
            # Compute discriminator loss
            d_loss = real_loss + fake_loss
            # Calculate gradients for discriminator
            d_loss.backward()
            # Clip gradients to avoid exploding gradients
            nn.utils.clip_grad_norm_(netD.parameters(), GRADIENT_CLIP)
            # Update discriminator parameters
            optimizerD.step()
            # -----------------
            #  Train Generator
            # -----------------
            netG.zero_grad()
            # Generate fake images
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_images = netG(noise, labels)
            # Classify fake images
            outputs = netD(fake_images, labels)
            # Compute generator loss
            g_loss = criterion(outputs, torch.ones_like(outputs))
            # Calculate gradients for generator
            g_loss.backward()
            # Clip gradients to avoid exploding gradients
            nn.utils.clip_grad_norm_(netG.parameters(), GRADIENT_CLIP)
            # Update generator parameters
            optimizerG.step()
        epoch_duration = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")
        if (epoch + 1) % PROGRESS_INTERVAL == 0:
            save_progress_samples(netG, epoch)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(epoch + 1, netG, netD, optimizerG, optimizerD)
            print(f"\nCheckpoint saved at epoch {epoch+1}")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    print("\nTraining completed successfully!")

def save_progress_samples(netG, epoch):
    with torch.no_grad():
        num_samples = 4
        noise = torch.randn(num_samples, LATENT_DIM, 1, 1, device=device)
        labels = torch.full((num_samples,), 4, dtype=torch.long, device=device)
        samples = (netG(noise, labels) + 1) / 2  # Scale to [0, 1]
        save_path = os.path.join(progress_dir, f"epoch_{epoch+1}_samples.png")
        save_image(samples, save_path, nrow=2, normalize=True)
        print(f"Saved progress samples to {save_path}")

def save_checkpoint(epoch, netG, netD, optimizerG, optimizerD):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
    }, checkpoint_path)

    # Keep only the latest 3 checkpoints
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")))
    for old_checkpoint in checkpoints[:-3]:
        os.remove(old_checkpoint)

# Main Function
def main():
    transform = initialize_environment()
    dataloader = create_data_loaders(transform)
    netG, netD, optimizerG, optimizerD, start_epoch = initialize_models()
    train(netG, netD, dataloader, optimizerG, optimizerD, start_epoch)

if __name__ == "__main__":
    main()