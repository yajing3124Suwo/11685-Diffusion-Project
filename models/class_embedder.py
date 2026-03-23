# -*- coding: utf-8 -*-
import torch.nn as nn


class ClassEmbedder(nn.Module):
    """Embeds class ids 0..n_classes-1; index num_classes is the null / unconditional token."""

    def __init__(self, dim, n_classes):
        super(ClassEmbedder, self).__init__()
        self.embedding = nn.Embedding(n_classes + 1, dim)
        self.num_classes = n_classes

    def forward(self, labels):
        return self.embedding(labels)
