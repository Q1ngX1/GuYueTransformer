import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedLoss(nn.Module):
    """
    Combines SmoothCrossEntropyLoss and Mean Squared Error loss.
    """
    __constants__ = ['label_smoothing', 'vocab_size', 'ignore_index', 'reduction', 'mse_weight', 'ce_weight']

    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, reduction='mean', is_logits=True, mse_weight=0.5,ce_weight=0.5):
        super(MixedLoss, self).__init__()
        assert 0.0 <= label_smoothing <= 1.0

        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.input_is_logits = is_logits
        self.mse_weight = mse_weight
        self.ce_weight = ce_weight

    def forward(self, input, target):
        batch_size, seq_len = input.shape[0], input.shape[1]
        mask = (target == self.ignore_index).unsqueeze(-1)

        # Convert target to one-hot encoding
        q = F.one_hot(target.long(), self.vocab_size).type(torch.float32)
        u = 1.0 / self.vocab_size
        q_prime = (1.0 - self.label_smoothing) * q + self.label_smoothing * u
        q_prime = q_prime.masked_fill(mask, 0)

        ce = self.cross_entropy_with_logits(q_prime, input)
        mse = F.mse_loss(input, q_prime.float(), reduction='none')
        mse = mse.sum(dim=-1)  # Sum over the vocabulary dimension
        mse = mse.masked_fill(mask.squeeze(-1), 0)  # Ignore masked positions

        if self.reduction == 'mean':
            lengths = torch.sum(target != self.ignore_index)
            ce = ce.sum() / lengths
            mse = mse.sum() / lengths
        elif self.reduction == 'sum':
            ce = ce.sum()
            mse = mse.sum()
        else:
            raise NotImplementedError

        return self.mse_weight * mse + self.ce_weight * ce

    def cross_entropy_with_logits(self, p, q):
        return -torch.sum(p * (q - q.logsumexp(dim=-1, keepdim=True)), dim=-1)