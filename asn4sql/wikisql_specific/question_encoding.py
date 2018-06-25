"""
Defines the question encoding for WikiSQL; see
WikiSQLSpecificModel for full docs.
"""

from absl import flags
import torch
from torch import nn

flags.DEFINE_integer(
    'sequence_question_encoding_size', 64,
    'hidden state size for the question sequential '
    'encoding; this should be even')
flags.DEFINE_integer(
    'question_encoding_size', 64, 'hidden state size for the whole-table '
    'question encoding')
flags.DEFINE_integer('ent_embedding_size', 32,
                     'embedding size for part-of-speech tags')


class QuestionEncoding(nn.Module):
    """
    The question encoding accepts both the src (natural language question)
    and ent (spacy part-of-speech annotations) entries, unembedded, from the
    WikiSQL example.

    It concatenates the inputs entrywise.

    Then a bidirectional LSTM provides a per-token encoding.

    Requires a pre-generated src encoding as input and the torchtext
    field for the part-of-speech encoding, which is learned from scratch.

    The sequence encoding size is in the sequence_size member,
    whereas final_size describes the encoding size for the final
    whole-question summary.
    """

    def __init__(self, src_embedding, ent_field):
        super().__init__()
        self.src_embedding = src_embedding
        num_words = len(ent_field.vocab.stoi)
        self.ent_embedding = nn.Embedding(num_words,
                                          flags.FLAGS.ent_embedding_size)

        self.sequence_size = flags.FLAGS.sequence_question_encoding_size
        self.final_size = flags.FLAGS.question_encoding_size

        assert self.sequence_size % 2 == 0, self.sequence_size

        embedding_dim = (self.src_embedding.embedding_dim +
                         self.ent_embedding.embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            self.sequence_size // 2,
            num_layers=1,
            bidirectional=True)

    def forward(self, src_seq_s, ent_seq_s):
        """
        Given the input question sequence of length s and its corresponding
        POS tags, both numbericalized, this method returns the sequence
        of question per-token encodings and the cumulative whole-question
        encoding vector.
        """
        src_seq_se = self.src_embedding(src_seq_s)
        ent_seq_se = self.ent_embedding(ent_seq_s)

        question_seq_se = torch.cat([src_seq_se, ent_seq_se], dim=1)
        output_seq_s1e, (hidden_21e, _) = self.lstm(
            question_seq_se.unsqueeze(1))
        output_seq_se = output_seq_s1e.squeeze(1)
        return output_seq_se, hidden_21e.view(-1)
