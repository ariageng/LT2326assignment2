​To run train.py with 10 epochs:  train.py -e 10
​To run autoencoder.py with 10 epochs: python autoencoder.py -e 10
​To run augmented_autoencoder.py with 10 epochs: python augmented_autoencoder.py -e 10


-Bonus A
​I changed conv2d to (3, 32, (3,3), padding=1), using smaller 3x3 window, more channels/pattern detectors (32).
​I also convert all images to 256*256 px
​Model (wikiart.pth) accuracy after these modifications:0.08253968507051468

