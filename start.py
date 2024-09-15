import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import jsonschema
import json
import sys
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from analysis import plot_class_distribution
from net_runner import NetRunner

def ifDataExist(directory):
    data_path = Path(directory)
    if not data_path.exists() or not data_path.is_dir():
        print(f"La directory {directory} non esiste o non è valida.")
        return True
    
    mat_files = list(data_path.glob('*.mat'))
    
    if len(mat_files) >= 2:
        print(f"Sono stati trovati {len(mat_files)} file .mat.")
        return False
    else:
        print(f"Sono stati trovati solo {len(mat_files)} file .mat.")
        return True
    
def main(cfg):
    
    DOWNLOAD = ifDataExist('./data')
    
    out_path = Path(cfg.io.out_path)
    if not out_path.exists():
        out_path.mkdir()
    
    transform = transforms.Compose([
        transforms.RandomRotation(10),  #Data Augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Caricamento del dataset SVHN
    train_set = torchvision.datasets.SVHN(root='./data', split='train', download=DOWNLOAD, transform=transform)
    test_set = torchvision.datasets.SVHN(root='./data', split='test', download=DOWNLOAD, transform=transform)
     
    # Plot della distribuzione delle classi
    #plot_class_distribution('./data/train_32x32.mat')

    # Riduzione del dataset se richiesto
    if cfg.config.reduce_dataset:
        train_indices, _ = train_test_split(range(len(train_set)), train_size=cfg.config.reduction_factor, stratify=train_set.labels)
        test_indices, _ = train_test_split(range(len(test_set)), train_size=cfg.config.reduction_factor, stratify=test_set.labels)
        
        train_set = Subset(train_set, train_indices)
        test_set = Subset(test_set, test_indices)

    train_indices, val_indices = train_test_split(range(len(train_set)), test_size=0.2, stratify=np.array([train_set[i][1] for i in range(len(train_set))])) 

    # Creazione dei nuovi Subset
    f_train_set = Subset(train_set, train_indices)
    val_set = Subset(train_set, val_indices)
    
    train_loader = DataLoader(f_train_set, batch_size=cfg.config.batch_size, shuffle=True) # Shuffle per evitare overfitting (mescolamento dei dati)
    val_loader = DataLoader(val_set, batch_size=cfg.config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg.config.batch_size, shuffle=False)

    netrunner = NetRunner(cfg)
    
    if cfg.config.training:
        print('Training...')
        netrunner.train(train_loader, val_loader)
        print('Testing...')
        netrunner.test(test_loader, use_current_model=True)
    else:
        print('Testing...')
        netrunner.test(test_loader)




if __name__ == '__main__':
    data_file, schema_file = Path('./config.json'), Path('./config_schema.json')
    valid_input = data_file.is_file() and schema_file.is_file() and data_file.suffix == '.json' and schema_file.suffix == '.json'
 
    if valid_input: # Validazione del config.json      
        with open(Path(data_file)) as d:
            with open(Path(schema_file)) as s:
                data, schema = json.load(d), json.load(s)
                
                try:
                    jsonschema.validate(instance=data, schema=schema)                    
                except jsonschema.exceptions.ValidationError:
                    print(f'Json config file is not following schema rules.')
                    sys.exit(-1)
                except jsonschema.exceptions.SchemaError:
                    print(f'Json config schema file is invalid.')
                    sys.exit(-1)

   
    with open(Path(data_file)) as d:
        main(json.loads(d.read(), object_hook=lambda d: SimpleNamespace(**d)))
    
    