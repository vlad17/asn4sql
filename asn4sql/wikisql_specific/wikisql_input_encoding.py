"""
Defines the encoding for both input sequences (the schema and question)
in any WikiSQL example point.
"""

import torch
from torch import nn

from .double_attention import IndependentAttention
from .nl_embedding import NLEmbedding
from .column_encoding import ColumnEncoding
from .question_encoding import QuestionEncoding


class WikiSQLInputEncoding(nn.Module):
    """
    Defines an encoding used for WikiSQL input sequences, along with
    independent attention over both sequence.
    """

    def __init__(self, fields, attend=True):
        super().__init__()
        natural_language_embedding = NLEmbedding(fields['src'])
        ent_field = fields['ent']
        tbl_field = fields['tbl']
        self.attend = attend
        self.question_encoding = QuestionEncoding(natural_language_embedding,
                                                  ent_field)
        self.column_encoding = ColumnEncoding(natural_language_embedding,
                                              tbl_field)
        self.question_seq_size = self.question_encoding.sequence_size
        self.column_seq_size = self.column_encoding.sequence_size
        if self.attend:
            self.attn = IndependentAttention(self.question_seq_size,
                                             self.column_seq_size, 0)
            self.attn_size = self.question_seq_size + self.column_seq_size

    def forward(self, ex):
        """
        Given a prepared query example, this outputs the sequence encodings
        for first the original question, then the schema, and finally the
        contextualized attention for both.
        """
        src_seq = self.question_encoding(ex['src'], ex['ent'])
        tbl_seq = self.column_encoding(ex['tbl'])
        if self.attend:
            attended = torch.cat(self.attn(src_seq, tbl_seq))
            return src_seq, tbl_seq, attended
        return src_seq, tbl_seq
