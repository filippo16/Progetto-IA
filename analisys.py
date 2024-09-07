import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

try:
    # Verifica se il file esiste
    file_path = './data/train_32x32.mat'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file {file_path} non esiste nella directory corrente.")

    # Caricamento del dataset SVHN
    print(f"Caricamento del file {file_path}...")
    data = sio.loadmat(file_path)

    # Verifica se 'y' è presente nel dizionario data
    if 'y' not in data:
        raise KeyError("La chiave 'y' non è presente nel file .mat")

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

    # Salvataggio del grafico
    plt.savefig('distribuzione_classi_svhn.png')
    print("Grafico salvato come 'distribuzione_classi_svhn.png'")

    # Stampa dei valori numerici
    for i in range(10):
        print(f"Classe {i}: {class_counts[i]} immagini")

except FileNotFoundError as e:
    print(f"Errore: {e}")
    print("Assicurati che il file .mat sia nella stessa directory dello script.")
except KeyError as e:
    print(f"Errore: {e}")
    print("Il file .mat non contiene i dati attesi. Verifica che sia il file corretto del dataset SVHN.")
except Exception as e:
    print(f"Si è verificato un errore imprevisto: {e}")
    import traceback
    traceback.print_exc()

print("Esecuzione dello script completata.")