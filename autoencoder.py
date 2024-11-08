import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
from wikiart import WikiArtDataset
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import argparse

class WikiAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder /compress
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, (3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            
            # Second conv block
            nn.Conv2d(64, 32, (3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            
            # Third conv block for more compression
            nn.Conv2d(32, 16, (3,3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
        
        # Decoder /reconstruct
        self.decoder = nn.Sequential(
            # First deconv block
            nn.ConvTranspose2d(16, 32, (3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            # Second deconv block
            nn.ConvTranspose2d(32, 64, (3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            # Final deconv block
            nn.ConvTranspose2d(64, 3, (3,3), padding=1),
            nn.Sigmoid(), 
            nn.Upsample(scale_factor=2)
        )


    def forward(self, image):
        encoded = self.encoder(image)
        decoded = self.decoder(encoded)

        return encoded, decoded 


def train(imgdir, device, epochs=10, batch_size=8, modelfile="wikiautoencoder.pth"):
    dataset = WikiArtDataset(imgdir, device, test_mode=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = WikiAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    print("Training...")
    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for batch in tqdm.tqdm(loader):
            images, _ = batch
            images = images.to(device) /255.0 # Normalize, each pixel values from 0 to 255
            
            optimizer.zero_grad()
            encoded, decoded = model(images)
            loss = criterion(decoded, images)
            loss.backward()
            optimizer.step()
            accumulate_loss += loss.item() 
        avg_loss = accumulate_loss / len(loader)
        print(f"In epoch {epoch}, average loss = {avg_loss:.6f}")

    torch.save(model.state_dict(), modelfile)
    return model

def cluster(model, imgdir, device):
    dataset = WikiArtDataset(imgdir, device, test_mode=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        images, _ = next(iter(loader)) # Get first batch of images for visualizing reconstruction sample
        images = images.to(device) / 255.0
        _, reconstructed = model(images)
        visualize_reconstruction(model, images, reconstructed, "final")

    # get encoded representations
    encodings = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in tqdm.tqdm(loader):
            images = images.to(device) 
            encoded, _ = model(images)
            # Flatten encoded representations
            encoded_flat = encoded.cpu().numpy().reshape(encoded.shape[0], -1)
            encodings.append(encoded_flat)
            labels.extend(batch_labels.numpy())

    # combine encodings
    encodings = np.concatenate(encodings, axis=0)
    labels = np.array(labels)
    
    # Use PCA to reduce to 2D
    pca = PCA(n_components=2)
    encodings_2d = pca.fit_transform(encodings)

    # Plot
    # Create a plot of 10x10 inches
    plt.figure(figsize=(12, 10))

    # Create scatter plot
    scatter = plt.scatter(
        encodings_2d[:, 0], # x-coordinates (first PCA component)
        encodings_2d[:, 1], # y-coordinates (second PCA component)
        c=labels,     # color by labels (art styles)
        cmap='hsv') # color map with 20 distinct colors
    plt.title('Art Style Clusters')
    plt.savefig('clusters.png')
    print("Plot saved as clusters.png")

def visualize_reconstruction(model, original, reconstructed, epoch):
    plt.figure(figsize=(15, 6))
    
    # Show 4 original images in first row
    for i in range(4):
        plt.subplot(2, 4, i+1)
        plt.imshow(original[i].cpu().detach().permute(1,2,0) /255.0)
        plt.axis('off')
        plt.title(f'Original {i+1}')
    
    # Show 4 reconstructed images in second row
    for i in range(4):
        plt.subplot(2, 4, i+5)
        plt.imshow(reconstructed[i].cpu().detach().permute(1,2,0)/255.0)
        plt.axis('off')
        plt.title(f'Reconstructed {i+1}')
    
    plt.tight_layout()
    plt.savefig(f'reconstruction_epoch_{epoch}.png')
    plt.close() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file", default="config.json")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")  # Add epochs argument
    args = parser.parse_args()
    
    config = json.load(open(args.config))
    imgdir = config["trainingdir"]
    device = config["device"]
    
    # Train
    model = train(imgdir, device, epochs=args.epochs)
    
    # Cluster
    cluster(model, imgdir, device)