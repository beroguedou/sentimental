import re
import os
import torch
import uuid
import argparse
import numpy as np
import pandas as pd
from config import *
from datetime import date
from data import SentimentalDataset, SentimentalDataLoader
from utils import train_fn, eval_fn
import transformers
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", dest="nb_epochs", type=int, default=50, help="maximum number of epochs to train the model.")
parser.add_argument("--batch", dest="batch_size", type=int, default=3, help="batch size for training the model.")
parser.add_argument("--workers", dest="num_workers", type=int, default=6, help="number of processors to transform the data.")
parser.add_argument("--stopping", dest="early_stopping", type=int, default=20,
                    help="number of epochs before stopping the model if no improvement.")
parser.add_argument("--name", dest="saving_name", type=str, default="sentimental_camembert_model.pth",
                    help="the name to give to the model when we want to save it.")
parser.add_argument("--device", type=str, default="cuda:0", help="determine which gpu to use (or cpu) to train the model.")

# Taking back the variables from the parser
train_args = parser.parse_args()
nb_epochs = train_args.nb_epochs
batch_size = train_args.batch_size
num_workers = train_args.num_workers
early_stopping = train_args.early_stopping
saving_name = train_args.saving_name 
device = train_args.device



# Creating training and validation datasets and dataloaders
train_dataset = SentimentalDataset(max_length, 
                                   dataset['train']['review'],
                                   dataset['train']['label'])

val_dataset = SentimentalDataset(max_length, 
                                 dataset['validation']['review'],
                                 dataset['validation']['label'])

train_dataloader = SentimentalDataLoader(train_dataset, 
                                               batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)

val_dataloader = SentimentalDataLoader(val_dataset,
                                             batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)


model.train().to(device)
# We don't want to lose quickly the pretrained weights so we put some part of the model to decay
parameters = list(model.named_parameters())
no_decay = ['classifier.dense.weight', 'classifier.dense.bias', 'bias', 'LayerNorm.bias', 'LayerNorm.weight',
            'classifier.out_proj.weight', 'classifier.out_proj.bias'] # We don't want any decay for them

optimizer_parameters = [
    {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay)], "weight_decay": 0.001},
    {"params": [p for n, p in parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

# Optimizer and criterion
optimizer = torch.optim.AdamW(params=optimizer_parameters, lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)

# Training loop !
count = 0
best_val_loss = np.Inf
for epoch in range(nb_epochs):
    train_fn(epoch, nb_epochs, model, train_dataloader, loss_fn, optimizer, device=device)
    val_loss = eval_fn(model,batch_size, val_dataloader, loss_fn, device)
    if val_loss < best_val_loss:
        count = 0
        best_val_loss = val_loss
        # Save the best checkpoint 
        torch.save(model.state_dict(), os.path.join('saved', saving_name))
    else:
        count += 1
    # Stop the training if the model doesn't improve anymore
    if count == early_stopping:
        print("******** EARLY STOPPING ********")
        break