import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
from wikiart import WikiArtDataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

class AugmentedAutoencoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 32, (3,3), padding=1)
        self.conv2 = nn.Conv2d(32, 16, (3,3), padding=1)
        self.maxpool = nn.MaxPool2d((2,2))
        self.relu = nn.ReLU()
        
        # Style embedding to 16-dimensional vector representation
        self.style_embedding = nn.Embedding(num_classes, 16)
        
        # Decoder with style
        self.deconv1 = nn.Conv2d(32, 32, (3,3), padding=1)  # 32 because 16+16 (image+style)
        self.deconv2 = nn.Sequential(
            nn.Conv2d(32, 3, (3, 3), padding=1),
            nn.Sigmoid(),  # Ensure output values are between [0, 1]
        )
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, style_idx):
        # Encode image
        x = x / 255.0
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        encoded = self.conv2(x)
        
        # Get art style embedding and reshape to match encoded feature map
        style = self.style_embedding(style_idx)
        style = style.view(-1, 16, 1, 1).expand(-1, -1, encoded.size(2), encoded.size(3))
        
        # Concatenate encoded image and style
        combined = torch.cat([encoded, style], dim=1)
        
        # Decode
        x = self.deconv1(combined)
        x = self.relu(x)
        x = self.upsample(x)
        decoded = self.deconv2(x)
        decoded = decoded * 255.0
        
        return decoded

def train_and_test(trainingdir, device, epochs=10):
    # Load dataset
    dataset = WikiArtDataset(trainingdir, device, test_mode=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = AugmentedAutoencoder(len(dataset.classes)).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Train
    print("Training...")
    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for images, labels in tqdm.tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(images, labels)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            accumulate_loss += loss.item() 
        avg_loss = accumulate_loss / len(loader)
        print(f"In epoch {epoch}, average loss = {avg_loss:.6f}")
    
    # Test with mismatched styles
    model.eval()
    print("\nTesting...")
    
    with torch.no_grad(): # disable gradient calculation
        # Get one test image
        test_image, original_style = next(iter(loader))
        test_image = test_image[0:1].to(device)  # Take first image
        
        # Original image
        plt.subplot(1, 4, 1)
        plt.imshow(test_image[0].cpu().permute(1, 2, 0) /255.0)
        plt.title(f'Original\nStyle: {dataset.classes[original_style[0]]}')
        
        # Try 3 different styles
        for i in range(3):
            new_style = torch.tensor([i]).to(device)
            generated = model(test_image, new_style)
            print("generated:",generated)
            
            plt.subplot(1, 4, i+2)
            plt.imshow(generated[0].cpu().permute(1, 2, 0)/255.0)
            plt.title(f'Generated\nStyle: {dataset.classes[i]}')
        
        plt.tight_layout()
        plt.savefig('style_transfer_results.png')
        print("Results saved as style_transfer_results.png")

if __name__ == "__main__":
    import json
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="configuration file", default="config.json")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    args = parser.parse_args()
    
    config = json.load(open(args.config))
    trainingdir = config["trainingdir"]
    device = config["device"]
    
    train_and_test(trainingdir, device, epochs=args.epochs)