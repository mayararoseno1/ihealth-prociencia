import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils.data_loader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

train_loader, test_loader = get_data_loaders(
    train_path='data/raw/training_data',
    test_path='data/raw/testing_data',
    batch_size=32
)

num_classes = len(train_loader.dataset.classes)
print(f"Número de classes: {num_classes}")

model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
model.num_classes = num_classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Época {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Acurácia: {accuracy:.2f}%")

os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/leukemia_model.pth')
print("✅ Modelo salvo em models/leukemia_model.pth")
