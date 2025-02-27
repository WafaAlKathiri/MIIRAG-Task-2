import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, num_classes=8, latent_dim=100, ngf=64, num_channels=3):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, noise, labels):
        label_embed = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, label_embed], 1)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, num_classes=8, ndf=64, num_channels=3):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, 1)  # Embed class labels

        self.model = nn.Sequential(
            nn.Conv2d(num_channels + 1, ndf, 4, 2, 1, bias=False),  # +1 for label embedding
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, images, labels):
        # Label embedding
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(images.size(0), 1, 1, 1)
        label_embedding = label_embedding.expand(images.size(0), 1, images.size(2), images.size(3))

        # Concatenate image with label embedding
        x = torch.cat([images, label_embedding], dim=1)
        return self.model(x).view(-1, 1)  # Flatten to [batch_size, 1]

# Example usage
if __name__ == '__main__':
    # Example usage
    batch_size = 4
    latent_dim = 100
    num_classes = 8
    img_size = 512

    # Generator test
    generator = Generator(num_classes=num_classes, latent_dim=latent_dim)
    noise = torch.randn(batch_size, latent_dim, 1, 1)
    labels = torch.randint(0, num_classes, (batch_size,))
    generated_images = generator(noise, labels)
    print("Generator output shape:", generated_images.shape)  # Expected: [4, 3, 512, 512]

    # Discriminator test
    discriminator = Discriminator(num_classes=num_classes)
    real_images = torch.randn(batch_size, 3, img_size, img_size)
    validity = discriminator(real_images, labels)
    print("Discriminator output shape:", validity.shape)  # Expected: [4, 1]