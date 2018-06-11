"""
Defines the question embedding for WikiSQL; see
WikiSQLSpecificModel for full docs.
"""

from absl import flags
import torch
from torch import nn

flags.DEFINE_integer('sequence_question_embedding_size', 256,
                     'hidden state size for the question sequential '
                     'embedding; this should be even')
flags.DEFINE_integer('question_embedding_size', 256,
                     'hidden state size for the whole-table '
                     'question embedding')
flags.DEFINE_integer('ent_embedding_size', 128,
                     'embedding size for part-of-speech tags')


class QuestionEmbedding(nn.Module):
    """
    The question embedding accepts both the src (natural language question)
    and ent (spacy part-of-speech annotations) entries, unembedded, from the
    WikiSQL example.

    It concatenates the inputs entrywise.

    Then a bidirectional LSTM provides a per-token embedding.
    A unidirectional LSTM then does a summary sweep to recover a summary
    of the entire question context.

    Requires a pre-generated src embedding as input and the torchtext
    field for the part-of-speech embedding, which is learned from scratch.

    The sequence embedding size is in the sequence_size member,
    whereas final_size describes the embedding size for the final
    whole-question summary.
    """

    def __init__(self, src_embedding, ent_field):
        self.src_embedding = src_embedding
        num_words = ent_field.vocab.vectors
        self.ent_embedding = nn.Embedding(
            num_words, flags.FLAGS.ent_embedding_size)

        self.sequence_size = flags.FLAG.sequence_question_embedding_size
        self.final_size = flags.FLAG.question_embedding_size

        assert self.sequence_size % 2 == 0, self.sequence_size

        self.lstm = nn.LSTM(
            self.embedding.embedding_dim, self.sequence_size // 2,
            num_layers=1, bidirectional=True)
        self.summary_lstm = nn.LSTM(self.sequence_size, self.final_size,
                                    num_layers=1, bidirectional=False)

    def forward(self, src_seq_s, ent_seq_s):
        """
        Given the input question sequence of length s and its corresponding
        POS tags, both numbericalized, this method returns the sequence
        of question per-token embeddings and the cumulative whole-question
        embedding vector.
        """
        src_seq_se = self.src_embedding(src_seq_s)
        ent_seq_se = self.ent_embedding(ent_seq_s)

        question_seq_se = torch.cat([src_seq_se, ent_seq_se], dim=1)
        output_seq_s1e, _ = self.lstm(question_seq_se.unsqueeze(1))
        output_seq_se = output_seq_s1e.squeeze(1)

        _, (final_hidden, _) = self.summary_lstm(output_seq_s1e)
        return output_seq_se, final_hidden.view(-1)
