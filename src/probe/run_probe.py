#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:03:11 2022

@author: José Antonio
"""

import torch
from tqdm import tqdm


#TODO EARLY STOPPING
def train_probe(train_loader, val_loader, probe, criterion, 
                lr = 0.001, epochs = 40):
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr) #lr=0.0005
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.1, patience=0)
    for epoch in tqdm(range(epochs), desc='[training epoch loop]'):
        probe.train()
        training_loss = 0.0
        for batch in tqdm(train_loader, desc='[training batch]'):
            #maybe align on the fly
            emb, dis, lens = batch
            # zero the parameter gradients
            optimizer.zero_grad()
            #compute outputs
            outputs = probe(emb)
            #loss = criterion(outputs, labels)
            loss, count = criterion(outputs, dis, lens)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss = training_loss / len(train_loader)
        eval_loss = eval_probe(val_loader, probe, criterion)    
        scheduler.step(eval_loss)
        tqdm.write('[epoch {}] Train loss: {}, Dev loss: {}'.format(epoch, 
                                                                training_loss, 
                                                                eval_loss))
    print('Finished Training')


def eval_probe(val_loader, probe, criterion):
    probe.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='[dev batch]'):
            probe.eval()
            emb, dis, lens = batch
            outputs = probe(emb)
            loss, count = criterion(outputs, dis, lens)
            eval_loss += loss.item()
    return eval_loss/len(val_loader)
