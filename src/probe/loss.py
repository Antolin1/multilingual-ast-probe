"""Custom loss classes for probing tasks."""

import torch
import torch.nn as nn


class L1DistanceLoss(nn.Module):
    """Custom L1 loss for distance matrices."""

    def __init__(self, device):
        super(L1DistanceLoss, self).__init__()
        self.device = device
        self.word_pair_dims = (1, 2)

    def forward(self, predictions, label_batch, length_batch):
        """ Computes L1 loss on distance matrices.

        Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the square of the sentence length)
        and then across the batch.

        Args:
          predictions: A pytorch batch of predicted distances
          label_batch: A pytorch batch of true distances
          length_batch: A pytorch batch of sentence lengths

        Returns:
          A tuple of:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
        """
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        total_sents = torch.sum((length_batch != 0)).float()
        squared_lengths = length_batch.pow(2).float()
        if total_sents > 0:
            loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_pair_dims)
            normalized_loss_per_sent = loss_per_sent / squared_lengths
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        else:
            batch_loss = torch.tensor(0.0, device=self.device)
        return batch_loss, total_sents


class L1DepthLoss(nn.Module):
    """Custom L1 loss for depth sequences."""

    def __init__(self, device):
        super(L1DepthLoss, self).__init__()
        self.device = device
        self.word_dim = 1

    def forward(self, predictions, label_batch, length_batch):
        """ Computes L1 loss on depth sequences.
        Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the sentence length)
        and then across the batch.
        Args:
          predictions: A pytorch batch of predicted depths
          label_batch: A pytorch batch of true depths
          length_batch: A pytorch batch of sentence lengths
        Returns:
          A tuple of:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
        """
        total_sents = torch.sum(length_batch != 0).float()
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        if total_sents > 0:
            loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_dim)
            normalized_loss_per_sent = loss_per_sent / length_batch.float()
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        else:
            batch_loss = torch.tensor(0.0, device=self.device)
        return batch_loss, total_sents


class ParserLoss(nn.Module):
    def __init__(self, loss='l1', pretrained=False):
        super(ParserLoss, self).__init__()
        self.cs = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss = loss
        self.pretrained = pretrained

    def forward(self, d_pred, scores_c, scores_u, d_real, c_real, u_real, length_batch,
                masks_c=None, masks_u=None):
        total_sents = torch.sum(length_batch != 0).float()
        labels_1s = (d_real != -1).float()
        d_pred_masked = d_pred * labels_1s  # b x seq-1
        d_real_masked = d_real * labels_1s  # b x seq-1
        if self.loss == 'l1':
            loss_d = torch.sum(torch.abs(d_pred_masked - d_real_masked), dim=1) / (length_batch.float() - 1)
            loss_d = torch.sum(loss_d) / total_sents
        elif self.loss == 'l2':
            loss_d = torch.sum(torch.abs(d_pred_masked - d_real_masked) ** 2, dim=1) / (length_batch.float() - 1)
            loss_d = torch.sum(loss_d) / total_sents
        elif self.loss == 'rank':
            lens_d = length_batch - 1
            max_len_d = torch.max(lens_d)
            mask = torch.arange(max_len_d, device=max_len_d.device)[None, :] < lens_d[:, None]
            loss_d = rankloss(d_pred, d_real, mask, exp=False)
        if masks_c is not None:
            scores_c += torch.unsqueeze(masks_c, dim=1)
        if masks_u is not None:
            scores_u += torch.unsqueeze(masks_u, dim=1)
        loss_c = self.cs(scores_c.view(-1, scores_c.shape[2]), c_real.view(-1))
        loss_u = self.cs(scores_u.view(-1, scores_u.shape[2]), u_real.view(-1))
        if self.pretrained:
            return loss_c + loss_u
        else:
            return loss_c + loss_d + loss_u


def rankloss(input, target, mask, exp=False):
    diff = input[:, :, None] - input[:, None, :]
    target_diff = ((target[:, :, None] - target[:, None, :]) > 0).float()
    mask = mask[:, :, None] * mask[:, None, :] * target_diff
    if exp:
        loss = torch.exp(torch.relu(target_diff - diff)) - 1
    else:
        loss = torch.relu(target_diff - diff)
    loss = (loss * mask).sum() / (mask.sum() + 1e-9)
    return loss
