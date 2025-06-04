# train_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils.data_loader import get_data_loaders

# Seleciona o dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carrega os dados
train_loader, test_loader = get_data_loaders(
    train_path='data/raw/training_data',
    test_path='data/raw/testing_data',
    batch_size=32
)

# Define número de classes com base nos dados
num_classes = len(train_loader.dataset.classes)
print(f"Número de classes: {num_classes}")

# Modelo base
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Época {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

# Salvar o modelo
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/leukemia_model.pth')
print("✅ Modelo salvo em models/leukemia_model.pth")
