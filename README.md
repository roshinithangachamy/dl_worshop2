## worshop2 - Building an AI Classifier: Identifying Cats, Dogs & Pandas with PyTorch
## Name: T.Roshini
## Reg no : 212223230175

## Aim:
To build an AI Classifier that identifies cats, dogs and pandas with PyTorch.

## Program:
```
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

```
<img width="376" height="104" alt="image" src="https://github.com/user-attachments/assets/da2c53dd-ea7d-4566-83c0-596d9541b988" />

```
from google.colab import files
uploaded = files.upload()  # browse your train_test.zip and upload
```
<img width="651" height="96" alt="image" src="https://github.com/user-attachments/assets/a34fc654-a446-432e-a785-11c1296a0128" />

```
!unzip /content/train_test.zip -d /content/data

```
<img width="787" height="564" alt="image" src="https://github.com/user-attachments/assets/ea96e53d-4ac0-4dcb-ba78-8a76ff67d1c6" />

```
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load dataset
train_data = datasets.ImageFolder(root="/content/data/train_test/Train", transform=train_transform)
test_data  = datasets.ImageFolder(root="/content/data/train_test/Test", transform=test_transform)

# DataLoader
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=2, shuffle=False)

print("Classes:", train_data.classes)  # ['cat', 'dog', 'panda']

```
<img width="388" height="73" alt="image" src="https://github.com/user-attachments/assets/7bd24ce5-420f-42e1-9acd-63f287fabb89" />

```
import torch
import torch.nn as nn
from torchvision import models

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load pre-trained ResNet18
model = models.resnet18(pretrained=True)

# Freeze all convolutional layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier (fc layer) for 3 classes
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 3)  # cat, dog, panda
)

# Move model to GPU if available
model = model.to(device)

print(model)
```

<img width="767" height="335" alt="image" src="https://github.com/user-attachments/assets/51bdafa2-8145-429e-a720-1d95120579f7" />

```
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
import torch
import torch.nn as nn
import torch.optim as optim

# Epochs
num_epochs = 5

best_acc = 0.0

for epoch in range(num_epochs):
    model.train()  # set model to training mode
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        # Move to GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # Backward pass + optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_data)
    epoch_acc = running_corrects.double() / len(train_data)

    print(f"Epoch {epoch+1}/{num_epochs} -> Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    # Save best model
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved!")
```

<img width="450" height="216" alt="image" src="https://github.com/user-attachments/assets/ecfa5dba-168d-4631-b699-84b7e7f13dff" />

```
# Load best model
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Test accuracy
running_corrects = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

test_acc = running_corrects.double() / len(test_data)
print("Test Accuracy:", test_acc.item())
```

<img width="396" height="65" alt="image" src="https://github.com/user-attachments/assets/ed29ff35-d9e7-4a0e-8ac2-6f99de18e1f1" />

```
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Collect all predictions & labels
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=train_data.classes)
disp.plot(cmap=plt.cm.Blues)
plt.show()
```
<img width="740" height="610" alt="image" src="https://github.com/user-attachments/assets/82992100-a34a-4a01-b2af-a36c974c3194" />

```




```

## Result:
Thus, build an AI Classifier that identifies cats, dogs and pandas with PyTorch has been done successfully.







