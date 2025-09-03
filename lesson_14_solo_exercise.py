# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 15:20:57 2025

@author: taske
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # TODO 1: Define model architecture with regularization layers
        # Should include: 2 conv layers, batch normalization after each conv,
        # 2 fully connected layers, dropout before final layer
        pass
    
    def forward(self, x):
        # TODO 2: Implement forward pass
        # Apply: conv -> bn -> relu -> pool -> conv -> bn -> relu -> pool
        # Then: flatten -> fc -> dropout -> fc
        pass

def get_transforms():
    # TODO 3: Create augmented and standard transforms
    # Train transform: should include data augmentation techniques
    # Test transform: only normalization, no augmentation
    train_transform = None
    test_transform = None
    return train_transform, test_transform

def get_data_loaders():
    # TODO 4: Setup CIFAR-10 data loaders
    # Use transforms from get_transforms(), set appropriate batch size
    # Return train_loader, test_loader
    pass

def label_smoothing_loss(outputs, targets, smoothing=0.1):
    # TODO 5: Implement label smoothing loss
    # Convert hard targets to soft targets, apply cross entropy
    # Should reduce overconfidence in predictions
    pass

class EarlyStopping:
    # TODO 6: Complete early stopping implementation
    # Should track: best validation loss, epochs without improvement
    # Methods needed: __init__(patience, min_delta), __call__(val_loss)
    # Returns True when training should stop
    pass

def train_epoch(model, loader, optimizer, device):
    # TODO 7: Implement one training epoch
    # Should: set model to train mode, iterate through batches,
    # compute loss, backprop, apply gradient clipping, update weights
    # Return: average loss, accuracy
    pass

def validate(model, loader, device):
    # TODO 8: Implement validation
    # Should: set model to eval mode, no gradient computation,
    # compute loss and accuracy over all batches
    # Return: average loss, accuracy
    pass

def setup_training(model):
    # TODO 9: Create optimizer and learning rate scheduler
    # Optimizer should include weight decay for L2 regularization
    # Scheduler should reduce LR when validation loss plateaus
    # Return: optimizer, scheduler
    pass

def plot_results(baseline_history, regularized_history):
    # TODO 10: Create comparison plots
    # Should show: training/validation loss curves, accuracy curves
    # Compare baseline vs regularized model performance
    pass

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # TODO 11: Initialize data loaders
    # Use get_data_loaders() function
    
    # TODO 12: Train baseline model
    # Create model without regularization, train for several epochs
    # Track training history (losses, accuracies)
    
    # TODO 13: Train regularized model  
    # Create model with all regularization techniques enabled
    # Use early stopping, label smoothing, all techniques covered
    
    # TODO 14: Compare results
    # Plot training curves, analyze which techniques helped most

if __name__ == "__main__":
    main()