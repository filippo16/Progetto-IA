import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from net import ResNet

class NetRunner:
    def __init__(self, optimizer, criterion, cfg, scheduler):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = ResNet(num_classes=10).to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.cfg = cfg

    def train(self, train_loader, val_loader, num_epochs):
        #self.model.train()
        
        es_start_epoch = self.cfg.config.early_stopping.start_epoch
        es_loss_evaluation_epochs = self.cfg.config.early_stopping.loss_evaluation_epochs
        es_patience = self.cfg.config.early_stopping.patience
        es_improvement_rate = self.cfg.config.early_stopping.improvement_rate
        
        best_tr_loss = float('inf')
        best_va_loss = float('inf')
        
        early_stop_check = False
        
        epoch_loss_values = []
        va_loss_no_improve = 0

        for epoch in range(num_epochs):
            running_loss = 0.0
            
            if (num_epochs + 1) == es_start_epoch: # Controllo per l'early stopping per inizio epoca
                early_stop_check = True
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                self.model.train()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                
                print(f'[Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}] Loss: {loss.item():.4f}')

            if scheduler:
                self.scheduler.step()
            
            
            avg_epoch_loss = running_loss / len(train_loader)
            epoch_loss_values.append(avg_epoch_loss)
            
            if avg_epoch_loss < best_tr_loss: # Salva il modello con la loss migliore
                best_tr_loss = avg_epoch_loss
                torch.save(self.model.state_dict(), './out/best_model_sd.pth')
                torch.save(self.model, './out/best_model.pth')
            
            if early_stop_check and (num_epochs + 1) % es_loss_evaluation_epochs == 0: # Per l'early stopping
                print('Validating...')
                val_loss = self.test(val_loader, use_current_model=True, validation=True)
                if va_loss < best_va_loss:
                    
                    # Calcolo il tasso di miglioramento.
                    improve_ratio = abs((va_loss / best_va_loss) - 1) * 100
                    
                    # Verifico che il miglioramento non sia inferiore al tasso.
                    if improve_ratio >= es_improvement_rate:
                        best_va_loss = va_loss
                        va_loss_no_improve = 0
                    else:
                        va_loss_no_improve += 1
                else:
                    va_loss_no_improve += 1
            
            if va_loss_no_improve >= es_patience: # Early stopping effettivo
                print(f"Early stopping all'epoca {epoch + 1}")
                break
            
            

        # Salva il modello  addestrato
        torch.save(self.model.state_dict(), './out/model_final_sd.pth') # Solo i parametri
        torch.save(self.model, './out/model_final.pth') # Intero modello
        print('Modello salvato')

        # Grafico loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), epoch_loss_values, marker='o', linestyle='-', color='r', label='Average Epoch Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss per Epoch')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


    def test(self, test_loader, use_current_model=False, validation=False):
        if use_current_model:
            mdoel = self.model
        else:
            model = ResNet(num_classes=10).to(self.device)
            model.load_state_dict(torch.load('./out/model_final_sd.pth')) # Creare una nuova rete?
        
       
        model.eval()

        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        average_loss = test_loss / len(test_loader)
        if validation:
            print(f'Validation set: Average loss: {average_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
            return average_loss
        else:
            print(f'Test set: Average loss: {average_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
        
