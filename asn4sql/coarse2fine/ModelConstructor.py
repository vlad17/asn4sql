"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""

from absl import flags
import torch.nn as nn
import torchtext.vocab

from . import modules
from .Models import (
    ParserModel, RNNEncoder, CondDecoder, TableRNNEncoder,
    MatchScorer, CondMatchScorer, CoAttention)
from ..data import wikisql
from ..utils import get_device

flags.DEFINE_integer('rnn_size', 250, 'number of RNN hidden units')
flags.DEFINE_integer('attn_hidden', 64, 'hidden units used for attention')
flags.DEFINE_enum('global_attention', 'general', ['dot', 'general', 'mlp'],
                  'the attention type to use: dotprot or general '
                  '(Luong) or MLP (Bahdanau)')
flags.DEFINE_enum('rnn_type', 'LSTM', ['LSTM', 'GRU'], 'RNN architecture')

def _padding_idx(vocab):
    return vocab.stoi[wikisql.PAD_WORD]

def _split_idx(vocab):
    return vocab.stoi[wikisql.SPLIT_WORD]

def make_word_embeddings(fields):
    # note: this trains the special tokens as well; need
    # the PartUpdateEmbedding if we don't want this
    num_words, word_vec_size = fields['src'].vocab.vectors.size()
    w_embeddings = nn.Embedding(num_words, word_vec_size,
                                padding_idx=_padding_idx(fields['src'].vocab))
    w_embeddings.weight.data.copy_(fields['src'].vocab.vectors)
    return w_embeddings


def make_embeddings(word_dict, vec_size):
    num_word = len(word_dict)
    w_embeddings = nn.Embedding(
        num_word, vec_size, padding_idx=_padding_idx(word_dict))
    return w_embeddings

def make_numerical_embedding(cardinality, vec_size):
    # we use the last element as the padding index for
    # pre-numericalized fields.
    w_embeddings = nn.Embedding(
        cardinality + 1, vec_size, padding_idx=cardinality)
    return w_embeddings


def make_encoder(embeddings, ent_embedding=None):
    enc_layers = 1
    dropout = 0.5
    lock_dropout = False
    weight_dropout = 0
    brnn = True # bidirectional
    return RNNEncoder(flags.FLAGS.rnn_type, brnn, enc_layers,
                      flags.FLAGS.rnn_size, dropout, lock_dropout,
                      weight_dropout, embeddings, ent_embedding)


def make_table_encoder(embeddings):
    split_type = 'incell' # 'outcell'
    merge_type = 'cat' # sub, mlp
    return TableRNNEncoder(make_encoder(embeddings), split_type, merge_type)


def make_cond_decoder(pad_index):
    input_size = flags.FLAGS.rnn_size
    dec_layers = 1
    dropout = 0.5
    lock_dropout = False
    weight_dropout = 0
    brnn = True # bidirectional
    return CondDecoder(flags.FLAGS.rnn_type, brnn, dec_layers, input_size,
                       flags.FLAGS.rnn_size, flags.FLAGS.global_attention,
                       flags.FLAGS.attn_hidden, dropout, lock_dropout,
                       weight_dropout, pad_index)



def make_co_attention(pad_index):
    brnn = True # bidirectional
    enc_layers = 1
    dropout = 0.5
    weight_dropout = 0
    return CoAttention(flags.FLAGS.rnn_type, brnn, enc_layers,
                       flags.FLAGS.rnn_size,
                       dropout, weight_dropout,
                       flags.FLAGS.global_attention,
                       flags.FLAGS.attn_hidden,
                       pad_index)


def build_model(fields):
    """
    Args:
        fields: `Field` objects for the model.
    Returns:
        the NMT-style model that coarse2fine is built after
    """

    # embedding
    w_embeddings = make_word_embeddings(fields)
    ent_embedding = make_embeddings(
        fields["ent"].vocab, 10)

    # Make question encoder.
    q_encoder = make_encoder(w_embeddings, ent_embedding)
    # Make table encoder.
    tbl_encoder = make_table_encoder(w_embeddings)
    co_attention = make_co_attention(_padding_idx(fields['src'].vocab))
    dropout = 0.5
    rnn_size = flags.FLAGS.rnn_size
    agg_classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(rnn_size, len(wikisql.AGGREGATION)),
        nn.LogSoftmax())
    score_size = 64
    sel_match = MatchScorer(2 * rnn_size, score_size, dropout)
    lay_classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(rnn_size, len(fields['lay'].vocab)),
        nn.LogSoftmax())

    # embedding
    # layout encoding
    cond_op_vec_size = 150
    cond_embedding = make_numerical_embedding(
        len(wikisql.CONDITIONAL), cond_op_vec_size)
    lay_encoder = make_encoder(cond_embedding)

    # Make cond models.
    cond_decoder = make_cond_decoder(len(wikisql.CONDITIONAL))
    cond_col_match = CondMatchScorer(
        MatchScorer(2 * rnn_size, score_size, dropout))
    cond_span_l_match = CondMatchScorer(
        MatchScorer(2 * rnn_size, score_size, dropout))
    cond_span_r_match = CondMatchScorer(
        MatchScorer(3 * rnn_size, score_size, dropout))

    # Make ParserModel
    pad_word_index = _padding_idx(fields['src'].vocab)
    model = ParserModel(q_encoder, tbl_encoder, co_attention, agg_classifier, sel_match, lay_classifier, cond_embedding,
                        lay_encoder, cond_decoder, cond_col_match, cond_span_l_match, cond_span_r_match, pad_word_index, _split_idx(fields['src'].vocab))

    # TODO checkpoint
    # if checkpoint is not None:
    #     print('Loading model')
    #     model.load_state_dict(checkpoint['model'])

    return model.to(get_device())
