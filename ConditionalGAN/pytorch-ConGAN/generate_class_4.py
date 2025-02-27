import torch
from torchvision.utils import save_image
from model import Generator
import os
import glob

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
latent_dim = 100  # MUST match training
num_classes = 8   # MUST match training
num_images = 100
class_label = 4   # Generate class 4 (cracks)
image_size = 512  # MUST match training
ngf = 64          # Generator feature map size (MUST match training)

# Paths
checkpoint_dir = "/kaggle/working/model_checkpoints"
save_path = "/kaggle/working/generated_crack_images"

# 1. Find the Latest Checkpoint
checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
if not checkpoints:
    raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
latest_checkpoint = max(checkpoints, key=os.path.getctime)
print(f"Using checkpoint: {latest_checkpoint}")

# 2. Initialize the Generator with the correct architecture
netG = Generator(
    num_classes=num_classes,
    latent_dim=latent_dim,
    ngf=ngf,
    num_channels=3
).to(device)

# 3. Load the Generator weights from the checkpoint
try:
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    print(f"Successfully loaded weights from {latest_checkpoint}")
except Exception as e:
    raise RuntimeError(f"Checkpoint loading failed: {str(e)}") from e

# 4. Set the Generator to evaluation mode
netG.eval()

# 5. Generate Images
os.makedirs(save_path, exist_ok=True)

with torch.no_grad():
    # Create noise vector
    noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
    # Create labels
    labels = torch.full((num_images,), class_label, dtype=torch.long, device=device)

    # Generate images
    fake_images = netG(noise, labels)

    # Normalize images
    fake_images = (fake_images + 1) / 2

    # Save images
    for i in range(num_images):
        save_image(
            fake_images[i],
            os.path.join(save_path, f"class4_crack_{i+1:04d}.png"),
            normalize=False # Images are already normalized
        )

    # Create and save grid
    grid_path = os.path.join(save_path, "class4_crack_grid.png")
    save_image(fake_images, grid_path, nrow=10, normalize=False)

print(f"\nGenerated {num_images} images to {save_path}")
print(f"Grid preview: {grid_path}")
