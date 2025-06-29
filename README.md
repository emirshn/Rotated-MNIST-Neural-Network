# Rotated-MNIST-Neural-Network
A PyTorch implementation of a neural network that jointly classifies digits (1, 2, 3, 4, 5, 7) and their orientations (0°, 90°, 180°, 270°) using a custom rotated MNIST dataset.

## Overview
This project:
- Augments the original MNIST dataset with rotated digits
- Excludes digit '0' and remaps labels to a custom 6-class problem
- Uses a simple CNN with shared convolutional layers and two output heads:
  - One for digit classification (6 classes)
  - One for rotation classification (4 classes)

## Features
- Combined loss for multitask learning: digit + rotation
- Predicts on custom images in the current directory
- 3-epoch training by default, with accuracy converging early
- Torch/torchvision-based and GPU-compatible

## Performance 
- **Digit Accuracy:** ~99.07%
- **Rotation Accuracy:** ~99.57%
> Tested on filtered MNIST digits with 4 rotation variants

## Results for MNIST
![image](https://github.com/user-attachments/assets/8b236a6c-a2de-4375-b85a-75957ef91fe7)

## Results for Custom Images
![image](https://github.com/user-attachments/assets/fda5f592-fa68-4477-ba4a-8ca64c5a361e)
