"""
Main SQLNet file, contains the end-to-end module encompassing all of its
components.

Xiaojun Xu, Chang Liu, Dawn Song. 2017. SQLNet: Generating Structured Queries
from Natural Language Without Reinforcement Learning.
"""

import torch.nn as nn
import torch.nn.functional as F


class SQLNet(nn.Module):
    """
    The SQLNet expects data to be of the WikiSQL format.

    It defines forward and backward operations for training the SQLNet.
    """

    def __init__(self):
        super().__init__()
        # just a placeholder implementation for now
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        return out
