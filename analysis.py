import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Caricamento del dataset SVHN
data = sio.loadmat('./data/train_32x32.mat')

# Estrazione e correzione delle etichette
labels = data['y'].flatten()
labels[labels == 10] = 0  # Corregge la classe '10' a '0'

# Conteggio delle classi
class_counts = Counter(labels)

# Preparazione dei dati per il grafico
classes = np.arange(10)
counts = [class_counts[i] for i in classes]

# Creazione del grafico
plt.figure(figsize=(10, 6))
plt.bar(classes, counts)
plt.xlabel('Classi (0-9)')
plt.ylabel('Numero di immagini')
plt.title('Distribuzione delle Classi nel Dataset SVHN')
plt.xticks(classes)
plt.show()

# Salvataggio del grafico
plt.savefig('distribuzione_classi_svhn.png')

# Stampa dei valori numerici
for i in range(10):
    print(f"Classe {i}: {class_counts[i]} immagini")