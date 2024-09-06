import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from net import ResNet
from net_runner import train, test

# Parametri configurabili
batch_size = 16
num_epochs = 15
learning_rate = 0.001
REDUCE_DATASET = True  # Imposta a True per ridurre il dataset
reduction_factor = 0.1  # Percentuale del dataset da utilizzare (es. 0.1 = 10%)
DOWNLOAD = False  # Imposta a False se il dataset è già stato scaricato
TRAINING = False  # Imposta a False per eseguire solo il test

# Trasformazioni per i dati di SVHN
transform = transforms.Compose([
    #transforms.RandomRotation(10),  #Data Augmentation
    #transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Caricamento del dataset SVHN
train_set = torchvision.datasets.SVHN(root='./data', split='train', download=DOWNLOAD, transform=transform)
test_set = torchvision.datasets.SVHN(root='./data', split='test', download=DOWNLOAD, transform=transform)

# Riduzione del dataset se richiesto
if REDUCE_DATASET:
    train_indices, _ = train_test_split(range(len(train_set)), train_size=reduction_factor, stratify=train_set.labels)
    test_indices, _ = train_test_split(range(len(test_set)), train_size=reduction_factor, stratify=test_set.labels)
    
    train_set = Subset(train_set, train_indices)
    test_set = Subset(test_set, test_indices)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modello
model = ResNet(num_classes=10).to(device)

# Ottimizzatore e criterio di perdita
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

if TRAINING:
    train(model, device, train_loader, optimizer, criterion, num_epochs, scheduler)
else:
    test(model, device, test_loader, criterion)
