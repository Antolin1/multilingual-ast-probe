# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, input_ids=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        outputs = (outputs * input_ids.ne(1)[:, :, None]).sum(1) / input_ids.ne(1).sum(-1)[:, None]
        return torch.nn.functional.normalize(outputs, p=2, dim=1)
