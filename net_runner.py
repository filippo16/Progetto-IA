import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torch
import matplotlib.pyplot as plt

def train(model, device, train_loader, optimizer, criterion, num_epochs, scheduler):
    model.train()
    
    # Lista per memorizzare la perdita media per ogni epoca
    epoch_loss_values = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Stampa della perdita per ogni batch
            print(f'[Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}] Loss: {loss.item():.4f}')

        if scheduler:
            scheduler.step()
        
        # Calcola e aggiungi la perdita media di quest'epoca
        avg_epoch_loss = running_loss / len(train_loader)
        epoch_loss_values.append(avg_epoch_loss)
        
    # Salva il modello addestrato
    torch.save(model.state_dict(), './out/model_final.pth')
    print('Modello salvato come ./out/model_final.pth')

    # Grafico della loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), epoch_loss_values, marker='o', linestyle='-', color='r', label='Average Epoch Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def test(model, device, test_loader, criterion):
    # Carica il modello salvato
    model.load_state_dict(torch.load('./out/model_final.pth'))
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    average_loss = test_loss / len(test_loader)
    
    print(f'Test set: Average loss: {average_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
