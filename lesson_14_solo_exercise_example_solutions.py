# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 15:23:34 2025

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
import numpy as np

class CNN(nn.Module):
    def __init__(self, num_classes=10, use_regularization=True):
        super(CNN, self).__init__()
        # Solution 1: Define model architecture with regularization layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_regularization else nn.Identity()
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_regularization else nn.Identity()
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5) if use_regularization else nn.Identity()
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Solution 2: Implement forward pass
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_transforms():
    # Solution 3: Create augmented and standard transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return train_transform, test_transform

def get_data_loaders():
    # Solution 4: Setup CIFAR-10 data loaders
    train_transform, test_transform = get_transforms()
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                               download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader

def label_smoothing_loss(outputs, targets, smoothing=0.1):
    # Solution 5: Implement label smoothing loss
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(outputs, dim=-1)
    targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets_smooth = targets_one_hot * confidence + smoothing / outputs.size(1)
    loss = -(targets_smooth * log_probs).sum(dim=1).mean()
    return loss

class EarlyStopping:
    # Solution 6: Complete early stopping implementation
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

def train_epoch(model, loader, optimizer, device, use_label_smoothing=False):
    # Solution 7: Implement one training epoch
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        
        if use_label_smoothing:
            loss = label_smoothing_loss(outputs, targets)
        else:
            loss = F.cross_entropy(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, device, use_label_smoothing=False):
    # Solution 8: Implement validation
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            
            if use_label_smoothing:
                loss = label_smoothing_loss(outputs, targets)
            else:
                loss = F.cross_entropy(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def setup_training(model, use_regularization=True):
    # Solution 9: Create optimizer and learning rate scheduler
    weight_decay = 1e-4 if use_regularization else 0
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    return optimizer, scheduler

def plot_results(baseline_history, regularized_history):
    # Solution 10: Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Training loss
    ax1.plot(baseline_history['train_loss'], label='Baseline', color='red')
    ax1.plot(regularized_history['train_loss'], label='Regularized', color='blue')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Validation loss
    ax2.plot(baseline_history['val_loss'], label='Baseline', color='red')
    ax2.plot(regularized_history['val_loss'], label='Regularized', color='blue')
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    # Training accuracy
    ax3.plot(baseline_history['train_acc'], label='Baseline', color='red')
    ax3.plot(regularized_history['train_acc'], label='Regularized', color='blue')
    ax3.set_title('Training Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    
    # Validation accuracy
    ax4.plot(baseline_history['val_acc'], label='Baseline', color='red')
    ax4.plot(regularized_history['val_acc'], label='Regularized', color='blue')
    ax4.set_title('Validation Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def train_model(model, train_loader, test_loader, device, epochs=15, use_regularization=True):
    optimizer, scheduler = setup_training(model, use_regularization)
    early_stopping = EarlyStopping(patience=5) if use_regularization else None
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, use_regularization)
        val_loss, val_acc = validate(model, test_loader, device, use_regularization)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if early_stopping and early_stopping(val_loss):
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return history

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Solution 11: Initialize data loaders
    train_loader, test_loader = get_data_loaders()
    
    # Solution 12: Train baseline model
    print("Training baseline model...")
    baseline_model = CNN(use_regularization=False).to(device)
    baseline_history = train_model(baseline_model, train_loader, test_loader, device, 
                                 epochs=15, use_regularization=False)
    
    # Solution 13: Train regularized model
    print("\nTraining regularized model...")
    regularized_model = CNN(use_regularization=True).to(device)
    regularized_history = train_model(regularized_model, train_loader, test_loader, device, 
                                    epochs=15, use_regularization=True)
    
    # Solution 14: Compare results
    plot_results(baseline_history, regularized_history)
    
    print(f"\nFinal Results:")
    print(f"Baseline - Val Acc: {baseline_history['val_acc'][-1]:.2f}%")
    print(f"Regularized - Val Acc: {regularized_history['val_acc'][-1]:.2f}%")

if __name__ == "__main__":
    main()