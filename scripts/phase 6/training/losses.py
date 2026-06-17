"""costruire la funzione di loss corretta per il problema,
tenendo conto dello sbilanciamento tra le classi.
Utilizzando una CrossEntropyLoss standard, il modello
privilegia le classi più frequenti (Falciparum).
Di conseguenza, assegnerò un peso maggiore alle classi rare"""

import numpy as np
import torch
import torch.nn as nn


def compute_class_weights(label_ids, num_classes=4):
    """Riceve in input:
    - label_ids -> array con tutti i label numerici del training set
    - num_classes -> numero di classi
    
    Conta quante volte ogni classe appare nel training set e
    calcola il peso come rapporto tra numero totale di campoioni
    e numero di campioni di quella classe:
    peso_classe_i = N_totale / (N_classi x N_campioni_classe_i)"""

    label_ids = np.array(label_ids)
    class_count = np.bincount(label_ids, minlength=num_classes)
    class_weights = len(label_ids) / (num_classes * class_count)


    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    for i,w in enumerate(weights_tensor):
        print(f"Class {i}: weights = {w:.2f}")

    return weights_tensor

def get_loss_function(label_ids, device, num_classes=4):
    weights = compute_class_weights(label_ids, num_classes)
    weights = weights.to(device)

    loss = nn.CrossEntropyLoss(weight=weights)
    return loss
