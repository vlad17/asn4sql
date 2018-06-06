"""
Main SQLNet file, contains the end-to-end module encompassing all of its
components.

Xiaojun Xu, Chang Liu, Dawn Song. 2017. SQLNet: Generating Structured Queries
from Natural Language Without Reinforcement Learning.
"""

import torch
import torch.nn as nn

from ..data import load_glove


class SQLNet(nn.Module):
    """
    The SQLNet expects data to be of the WikiSQL format.

    It defines forward and backward operations for training the SQLNet.

    It assumes the "oracle sketch" is given; i.e., that the SQL
    statements are of the restricted WikiSQL form.
    """

    def __init__(self):
        super().__init__()
        self.word_to_idx, embed = load_glove()

        # TODO: unfreeze (what about UNKNOWN?)
        # TODO: use separate agg / sel / cond embed
        # TODO: sparse grads? keep on CPU?
        embed = torch.from_numpy(embed)
        self.embedding = nn.Embedding.from_pretrained(embed, freeze=True)

        # self.agg_pred = AggPredictor()
        # self.sel_pred = SelPredictor()
        # self.cond_pred = CondPredictor()

        # # just a placeholder implementation for now
        # self.fc1 = nn.Linear(10, 10)
        # self.fc2 = nn.Linear(10, 10)

    def forward(self, questions, _columns):
        """
        Given a batch of WikiSQL questions (as a list of tokens)
        and descriptions of the columns (list of lists of tokens),
        output the tuple of batches
        (agg_idx, sel_idx, conds),
        where
        agg_idx are indices of Query.AGGREGATION,
        sel_idx are indices into the column length,
        conds
        aggregation index (based on Query.AGGREGATION)
        and the conditional index (based on Query.CONDITIONAL)
        """
        n = len(questions)
        m = max(len(q) for q in questions)
        # n = batch size
        # m = max question length this batch
        # e = embed dim
        tok_idxs_nm = torch.zeros(n, m, dtype=torch.int32)
        for i, question in enumerate(questions):
            # should really build a mask here for unknown stuff (during and
            # after the question tokens)...
            question_len = len(question)
            tok_idxs_nm[i, :question_len] = [
                self.word_to_idx.get(s.token, 0) for s in question
            ]
        question_embedding_nme = self.embedding(tok_idxs_nm)
        # packed sequence stuff...
        return question_embedding_nme
