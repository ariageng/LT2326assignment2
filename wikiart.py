import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F # image tranform
from torch.optim import Adam
import tqdm #progress bar
import random

class WikiArtImage:
    def __init__(self, imgdir, label, filename):
        self.imgdir = imgdir
        self.label = label
        self.filename = filename
        self.image = None
        self.loaded = False

    def get(self):
        if not self.loaded:
            self.image = read_image(os.path.join(self.imgdir, self.label, self.filename)).float()
            self.loaded = True

        return self.image

class WikiArtDataset(Dataset):
    def __init__(self, imgdir, device="cpu", test_mode=False):
        walking = os.walk(imgdir)
        filedict = {}
        indices = []
        classes = set()
        print("Gathering files for {}".format(imgdir))
        for item in walking:
            sys.stdout.write('.')
            arttype = os.path.basename(item[0])
            artfiles = item[2]
            for art in artfiles:
                filedict[art] = WikiArtImage(imgdir, arttype, art)
                indices.append(art)
                classes.add(arttype)
        print("...finished")

        # Count the amount of the images for each class
        class_counts = {}
        for art in filedict.values():
            if art.label not in class_counts:
                class_counts[art.label] = 0
            class_counts[art.label] += 1


        if not test_mode: # only undersample when training
        
            print("Before undersampling:")
            for label, count in class_counts.items():
                print(f"{label}: {count}")
            # Get the class with fewest data
            min_class = min(class_counts.values())
            print("Minimal amount of all classes:", min_class)

            # Undersample
            undersampled_filedict = {}
            undersampled_indices = []
            
            random.seed(400)
            for label in classes:
                class_images = [name for name in filedict.keys() 
                            if filedict[name].label == label]
                selected = random.sample(class_images, min_class)
                for img in selected:
                    undersampled_filedict[img] = filedict[img]
                    undersampled_indices.append(img)

            self.filedict = undersampled_filedict
            self.indices = undersampled_indices 
        else:
            self.filedict = filedict
            self.indices = indices
    
        self.device = device
        self.imgdir = imgdir
        self.classes = list(classes)
        
    def __len__(self):
        return len(self.filedict)

    def __getitem__(self, idx):
        imgname = self.indices[idx]
        imgobj = self.filedict[imgname]
        ilabel = self.classes.index(imgobj.label)
        image = imgobj.get().to(self.device)

        # make sure all images resized to 256x256
        image = F.resize(image, [256, 256])

        return image, ilabel

class WikiArtModel(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()

        self.conv2d = nn.Conv2d(3, 32, (3,3), padding=1) # 3 channels from "RGB", output 32 channels as "features" or "pattern detectors", 3x3 kernel size
        self.maxpool2d = nn.MaxPool2d((2,2), padding=1)
        self.flatten = nn.Flatten()
        self.batchnorm1d = nn.BatchNorm1d(532512)
        self.linear1 = nn.Linear(532512, 300)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(300, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, image):
        output = self.conv2d(image)
        #print("convout {}".format(output.size()))
        output = self.maxpool2d(output)
        #print("poolout {}".format(output.size()))        
        output = self.flatten(output)
        output = self.batchnorm1d(output)
        #print("poolout {}".format(output.size()))        
        output = self.linear1(output)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.linear2(output)
        return self.softmax(output)