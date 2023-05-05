import torch
import torch.nn as nn
import torchvision.models as models

# Charger le modèle Swin Transformer préentraîné sur ImageNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.swin_t(device)

# Changer la dernière couche pour qu'elle corresponde au nombre de classes de notre tâche
num_classes = 10
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# Charger les données d'entraînement et de validation
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Définir la fonction de coût et l'optimiseur
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005,)

# Fine-tuner le modèle
num_epochs = 800
for epoch in range(num_epochs):
    # Entraînement
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Évaluation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
    
    # Afficher les résultats de l'époque
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

    import torch
import torchvision.models as models

class SwinTModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(SwinTModel, self).__init__()
        # Chargez le modèle Swin Transformer avec des poids pré-entraînés
        self.model = models.swin_t(pretrained=True)
        
        # Remplacez la couche de classification existante par une nouvelle couche de classification adaptée au nombre de classes souhaité
        self.num_classes = num_classes
        in_features = self.model.head.in_features
        self.model.head = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    

""""""""" Swin Transformers kaggle  """

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os



import torch
import torchvision
from torchvision import datasets
from torchvision import transforms as T # for simplifying the transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models

## Now, we import timm, torchvision image models
import timm
from timm.loss import LabelSmoothingCrossEntropy # This is better than normal nn.CrossEntropyLoss



