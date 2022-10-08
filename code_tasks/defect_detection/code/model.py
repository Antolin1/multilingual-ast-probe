# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss

    
    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        if args.model_type == 't5':
            self.out_linear=nn.Linear(768, 1)
            self.dropout=nn.Dropout(p=0.1)
    
        
    def forward(self, input_ids=None,labels=None):
        if self.args.model_type != 't5': 
            outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        else:
            outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0][:,0,:]
            outputs=self.out_linear(self.dropout(outputs))
        logits=outputs
        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob
      
        
 
