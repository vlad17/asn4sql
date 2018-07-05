"""
This module procures data loaders for various datasets.
Inspired by the corresponding coarse2fine data preprocessing script.

https://github.com/donglixp/coarse2fine

Note datasets are cached.

General WikiSQL info:

WikiSQL has several forms of the same string as preprocessing. There is
the original string and the whitespace or punctuation that it has after it,
and there's the token of the original string extracted from StanfordCoreNLP.

An example from the WikiSQL dataset contains a natural language string
as well as schema information.
Note that WikiSQL has a very simplified version of SQL, so its
predicates all have a (potentially null) aggregation,
single selection column, and possibly multiple binary
conditional statements (all in conjunction).
Thus, all queries associated with a string are
always of the form

SELECT <agg>? col<digit>
  (WHERE (col<digit> <op> <literal parameter>*)
    (AND col<digit> <op> <literal parameter>*)*)?

<agg> is an aggregation from the AGGREGATION list
<op> is a binary operator from the CONDITIONAL list
The actual queries use parameterized SQL (they don't fill in the
literal parameters with string interpolation directly).

For more informative, diagnostic-only queries, we instead print queries
with the column description for column <i> instead of the true column name,
col<i>, and we interpolate the literals used to parameterize the queries.

A condition goes after a WHERE predicate.
In WikiSQL, all conditions are of the form <column> <op> <literal>,
e.g., person_name=sam.
CONDITIONAL describes all possible ops.
The column index (corresponding to a table schema referenced by the query
containing a condition)
The literal is always assumed to be a span of the input question
(the few examples where this is not the case are dropped).
"""

import collections
import copy
import os
import itertools
import functools
import json
import multiprocessing as mp
import locale

from absl import flags
from dateutil.parser import parse as dateparser
import numpy as np
from tqdm import tqdm
import track
import spacy
import textacy
import torch
import torchtext.vocab

from .fetch import check_or_fetch
from ..dbengine import DBEngine

AGGREGATION = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
CONDITIONAL = ['=', '>', '<']
SPLIT_WORD = '<|>'
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
SOME_INT_WORD = '<int>'
SOME_FLOAT_WORD = '<float>'
SOME_URL_WORD = '<url>'
SOME_EMAIL_WORD = '<email>'
SOME_DATE_WORD = '<int>'
SOME_NUMPREFIX_WORD = '<numprefix>'
SPECIALS = [
    UNK_WORD, SPLIT_WORD, PAD_WORD, SOME_INT_WORD, SOME_URL_WORD,
    SOME_EMAIL_WORD, SOME_DATE_WORD, SOME_NUMPREFIX_WORD
]

locale.setlocale(locale.LC_ALL, 'en_US.UTF8')  # USA USA USA


def _wikisql_data_readers(db):
    """
    Defines the fields, parsers, and validators for WikiSQL entries in one
    place. Note that to get the schema we need to open up a database
    connection, which is done here.

    See inline comments for a description of each column,

    src, ent, sel, table_id, tbl, cond_op, cond_col, cond_span_l,
    cond_span_r, original, after, agg, tbl_original

    Returns the dicitionary keyed by column name for the
    torchtext fields, parsers (query json -> parsed data), and
    validators (query json, parsed example row -> unit).
    """
    fields = {}
    parsers = {}
    validators = {}

    # src is a sequence of StanfordCoreNLP-tokenized lowercased words for the
    # natural language question being asked in the dataset
    def _parse_src(query_json):
        return query_json['question']['tok']

    field_src = torchtext.data.Field(
        batch_first=True, tokenize=_tokenize, pad_token=PAD_WORD)

    def _validate_src(_query_json, _ex):
        pass

    parsers['src'] = _parse_src
    fields['src'] = field_src
    validators['src'] = _validate_src

    # ent is a sequence of strings from a small vocab corresponding to src
    # strings in the same sequence index which identify the part of speech

    def _parse_ent(query_json):
        return query_json['question']['ent']

    field_ent = torchtext.data.Field(
        batch_first=True, tokenize=_tokenize, pad_token=PAD_WORD)

    def _validate_ent(_query_json, ex):
        if len(ex.src) != len(ex.ent):
            raise _QueryParseException('src length {} != ent length {}'.format(
                len(ex.src), len(ex.ent)))

    parsers['ent'] = _parse_ent
    fields['ent'] = field_ent
    validators['ent'] = _validate_ent

    # agg is the aggregation index, which must index into AGGREGATION
    def _parse_agg(query_json):
        return query_json['query']['agg']

    field_agg = torchtext.data.Field(
        tokenize=_tokenize,
        sequential=False,
        use_vocab=False,
        batch_first=True)

    def _validate_agg(_query_json, ex):
        if ex.agg < 0 or ex.agg >= len(AGGREGATION):
            raise _QueryParseException(
                'aggregation index {} not in [0, {})'.format(
                    ex.agg, len(AGGREGATION)))

    parsers['agg'] = _parse_agg
    fields['agg'] = field_agg
    validators['agg'] = _validate_agg

    # sel is the aggregation index, which must index into the query's
    # columns
    def _parse_sel(query_json):
        return query_json['query']['sel']

    field_sel = torchtext.data.Field(
        tokenize=_tokenize,
        sequential=False,
        use_vocab=False,
        batch_first=True)

    def _validate_sel(_query_json, ex):
        num_cols = len(db.get_schema(ex.table_id))
        if ex.sel < 0 or ex.sel >= num_cols:
            raise _QueryParseException(
                'selection index {} not in [0, num_columns={})'.format(
                    ex.sel, num_cols))

    parsers['sel'] = _parse_sel
    fields['sel'] = field_sel
    validators['sel'] = _validate_sel

    # table_id is the string table id (not meaningful; only used for
    # recovering query execution information)
    def _parse_table_id(query_json):
        table_id = query_json['table_id']
        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))
        return table_id

    field_table_id = torchtext.data.Field(
        tokenize=_tokenize, sequential=False, use_vocab=True, batch_first=True)

    def _validate_table_id(_query_json, ex):
        try:
            db.get_schema(ex.table_id)
        except Exception as e:
            raise _QueryParseException('error loading table: {}'.format(
                str(e)))

    parsers['table_id'] = _parse_table_id
    fields['table_id'] = field_table_id
    validators['table_id'] = _validate_table_id

    # tbl_original is the list of full column descriptions, which may
    # consist of multiple tokens
    def _parse_tbl_original(query_json):
        flat_cols = []
        for col_desc in query_json['table']['header']:
            original = col_desc['gloss']
            after = col_desc['after']
            colname = detokenize(original, after, drop_last=True)
            flat_cols.append(colname)
        return flat_cols

    field_tbl_original = torchtext.data.Field(
        batch_first=True, tokenize=_tokenize, pad_token=PAD_WORD)

    def _validate_tbl_original(_query_json, _ex):
        pass

    parsers['tbl_original'] = _parse_tbl_original
    fields['tbl_original'] = field_tbl_original
    validators['tbl_original'] = _validate_tbl_original

    # tbl gives the multi-word column descriptions for the columns
    # in the schema, which are separated by the SPLIT_WORD; the descriptions
    # are tokenized and lowercased with StanfordCoreNLP
    def _parse_tbl(query_json):
        flat_cols = []
        for col_desc in query_json['table']['header']:
            flat_cols.extend(col_desc['words'])
            flat_cols.append(SPLIT_WORD)
        # note extra split at the end... not my standard! from coarse2fine
        return flat_cols

    field_tbl = torchtext.data.Field(
        batch_first=True, tokenize=_tokenize, pad_token=PAD_WORD)

    def _validate_tbl(_query_json, ex):
        if not ex.tbl:
            raise _QueryParseException(
                "require at least one column in each query's table")
        num_cols = ex.tbl.count(SPLIT_WORD)
        schema_num_cols = len(db.get_schema(ex.table_id))
        if num_cols != schema_num_cols:
            raise _QueryParseException(
                'num columns in header {} != num columns in schema {}'.format(
                    num_cols, schema_num_cols))

    parsers['tbl'] = _parse_tbl
    fields['tbl'] = field_tbl
    validators['tbl'] = _validate_tbl

    # cond_op is the list of the conditional operation indices (unlike lay,
    # which is the concatenation of their string values)
    # note since it's a numerical sequence it has a -1 pad token.
    def _parse_cond_op(query_json):
        return [op_idx for _, op_idx, _ in query_json['query']['conds']]

    field_cond_op = torchtext.data.Field(
        tokenize=_tokenize, pad_token=-1, use_vocab=False, batch_first=True)

    def _validate_cond_op(_query_json, ex):
        for op_idx in ex.cond_op:
            if op_idx < 0 or op_idx >= len(CONDITIONAL):
                raise _QueryParseException(
                    'op index {} not valid'.format(op_idx))

    parsers['cond_op'] = _parse_cond_op
    fields['cond_op'] = field_cond_op
    validators['cond_op'] = _validate_cond_op

    # cond_col is the list of the column indices for the corresponding filter
    # coarse2fine relies on the pad token being 0 here and below, ugh...
    def _parse_cond_col(query_json):
        return [col_idx for col_idx, _, _ in query_json['query']['conds']]

    field_cond_col = torchtext.data.Field(
        batch_first=True, tokenize=_tokenize, pad_token=-1, use_vocab=False)

    def _validate_cond_col(_query_json, ex):
        num_cols = len(db.get_schema(ex.table_id))
        for col_idx in ex.cond_col:
            if col_idx < 0 or col_idx >= num_cols:
                raise _QueryParseException(
                    'column index {} does not match num columns in schema {}'
                    .format(col_idx, num_cols))

    parsers['cond_col'] = _parse_cond_col
    fields['cond_col'] = field_cond_col
    validators['cond_col'] = _validate_cond_col

    # cond_span_l is the list of the indices refering to the begining of the
    # natural language query that
    def _parse_cond_span_l(query_json):
        cond = _parse_span(query_json)
        return [l for l, _ in cond]

    field_cond_span_l = torchtext.data.Field(
        batch_first=True, tokenize=_tokenize, pad_token=-1, use_vocab=False)

    def _validate_cond_span_l(_query_json, _ex):
        pass

    parsers['cond_span_l'] = _parse_cond_span_l
    fields['cond_span_l'] = field_cond_span_l
    validators['cond_span_l'] = _validate_cond_span_l

    # cond_span_r corresponds to the end index of the literal span
    def _parse_cond_span_r(query_json):
        cond = _parse_span(query_json)
        return [r for _, r in cond]

    field_cond_span_r = torchtext.data.Field(
        batch_first=True, tokenize=_tokenize, pad_token=-1, use_vocab=False)

    def _validate_cond_span_r(_query_json, _ex):
        pass

    parsers['cond_span_r'] = _parse_cond_span_r
    fields['cond_span_r'] = field_cond_span_r
    validators['cond_span_r'] = _validate_cond_span_r

    # original is a sequence of strings from a corresponding to src
    # strings that are unprocessed
    def _parse_original(query_json):
        return query_json['question']['original']

    field_original = torchtext.data.Field(
        batch_first=True, tokenize=_tokenize, pad_token=PAD_WORD)

    def _validate_original(_query_json, ex):
        if len(ex.src) != len(ex.original):
            raise _QueryParseException('src length {} != original length {}'
                                       .format(len(ex.src), len(ex.original)))

    parsers['original'] = _parse_original
    fields['original'] = field_original
    validators['original'] = _validate_original

    return parsers, fields, validators


_URL = 'https://drive.google.com/uc?id=1uMG4cjaQGNU_HRW-2TPB5g904FITKuKE'


def wikisql(toy):
    """
    Loads the WikiSQL dataset. Per the original SQLNet implementation,
    a subsampled version is returned if the toy argument is true.

    Returns the train, val, and test TableDatasets for WikiSQL.
    """
    # compressed WikiSQL file after the annotation recommended in
    # https://github.com/salesforce/WikiSQL has been run.
    wikisql_dir = check_or_fetch('wikisql', 'wikisql.tgz', _URL)
    wikisql_dir = os.path.join(wikisql_dir, 'wikisql')

    track.debug('loading spacy tagger')
    _nlp()
    track.debug('loading vocabulary')
    pretrained_vocab(toy)

    train = _load_wikisql_data(
        os.path.join(wikisql_dir, 'annotated', 'train.jsonl'),
        os.path.join(wikisql_dir, 'dbs', 'train.db'), toy)
    val = _load_wikisql_data(
        os.path.join(wikisql_dir, 'annotated', 'dev.jsonl'),
        os.path.join(wikisql_dir, 'dbs', 'dev.db'), toy)
    test = _load_wikisql_data(
        os.path.join(wikisql_dir, 'annotated', 'test.jsonl'),
        os.path.join(wikisql_dir, 'dbs', 'test.db'), toy)

    return train, val, test


class Prediction:
    """
    Minimal container of data necessary to make a prediction about what a
    WikiSQL query will look like.
    """

    def __init__(self, ops, cols, span_ls, span_rs, agg, sel):
        self.ops = ops
        self.num_conds = len(ops)
        self.cols = cols
        self.span_ls = span_ls
        self.span_rs = span_rs
        self.agg = agg
        self.sel = sel
        # TODO validation ops cols, spans all same length
        # TODO more validation...
        # span_rs = [max(l, r) for l, r in zip(self.span_ls, self.span_rs)]

    def as_query_example(self, ex):
        """return a runnable instance of this prediction"""

        ex = copy.copy(ex)
        ex.cond_col = self.cols
        ex.sel = self.sel
        ex.cond_op = self.ops
        ex.cond_span_l = self.span_ls
        ex.cond_span_r = self.span_rs
        ex.agg = self.agg
        return ex

    def condition_logical_match(self, ex):
        """
        returns whether the conditions logically match with the given
        example.
        """
        if len(ex.cond_op) != self.num_conds:
            return False

        if not self.num_conds:
            return True

        # the condition can be fully represented as a 4-by-(num conds)
        # integer matrix
        true_cond = np.array(
            [ex.cond_col, ex.cond_op, ex.cond_span_l, ex.cond_span_r],
            dtype=int)
        pred_cond = np.array(
            [self.cols, self.ops, self.span_ls, self.span_rs], dtype=int)

        idx = np.lexsort(true_cond)
        true_cond = true_cond[:, idx]
        idx = np.lexsort(pred_cond)
        pred_cond = pred_cond[:, idx]

        return np.array_equal(true_cond, pred_cond)


class TableDataset(torchtext.data.Dataset):
    """WikiSQL dataset in torchtext format; pickleable."""

    def __init__(self, examples, fields, db):
        self.db_path = db.db_path
        self.db_engine = db
        self._toy_vocab = False
        super().__init__(examples, fields)

    @staticmethod
    def question(ex):
        """Print the question associated with a query"""
        return ''.join(ex.original)

    def build_vocab(self, max_size, toy, other_datasets):
        """
        Build the vocab form pretrained sources.
        """
        self._toy_vocab = toy
        self.fields['ent'].build_vocab(self, max_size=max_size, min_freq=0)

        # Need to keep words used in the dev and test tables as well; else
        # the GloVe pretrained values for those words will get thrown away.
        # This isn't cheating because we won't ever train on words in the
        # test set but not in the training set.
        shared_fields = ['src', 'tbl']
        shared_datasets = [self] + other_datasets
        shared_vocabs = []
        for field in shared_fields:
            for dataset in shared_datasets:
                dataset.fields[field].build_vocab(
                    dataset, max_size=max_size, min_freq=0)
                shared_vocabs.append(dataset.fields[field].vocab)
        pretrained = pretrained_vocab(toy)
        merged_vocab = _merge_vocabs(shared_vocabs, pretrained)
        for field in shared_fields:
            self.fields[field].vocab = merged_vocab

    def __getstate__(self):
        vocabs = {
            name: field.vocab
            for name, field in self.fields.items() if hasattr(field, 'vocab')
        }

        return {
            'examples': self.examples,
            'db_path': self.db_path,
            'vocabs': vocabs
        }

    def __setstate__(self, state):
        vocabs = state['vocabs']
        del state['vocabs']
        self.__dict__.update(state)
        self.db_engine = DBEngine(self.db_path)
        _, self.fields, _ = _wikisql_data_readers(self.db_engine)
        for field_name, vocab in vocabs.items():
            self.fields[field_name].vocab = vocab


def _count_lines(fname):
    with open(fname, 'r') as f:
        return sum(1 for line in f)


def _load_wikisql_data(jsonl_path, db_path, toy):
    # NOTE: because the tokenizer was StanfordCoreNLP and not spacy,
    # it's a bit finicky to use the spacy entity recognition in
    # order to describe the part of speech of all the query questions.
    # For this reason a slow python loop needs to be used to parse the ent
    # field of the dataset.

    queries = []
    db = DBEngine(db_path)

    parsers, fields, validators = _wikisql_data_readers(db)
    # weird api for Example.fromdict
    ex_fields = {k: [(k, v)] for k, v in fields.items()}

    excs = []
    for query_json in _annotate_queries(jsonl_path, toy):
        try:
            parsed_fields = {
                k: parse(query_json)
                for k, parse in parsers.items()
            }
            ex = torchtext.data.Example.fromdict(parsed_fields, ex_fields)
            for v in validators.values():
                v(query_json, ex)
            queries.append(ex)
        except _QueryParseException as e:
            excs.append(e.args[0])
    track.debug('dropped {} of {} queries{}{}', len(excs),
                len(excs) + len(queries), ':\n    '
                if excs else '', '\n    '.join(excs))

    return TableDataset(queries, fields, db)


def _find_sublist(haystack, needle):
    # the haystack is a list of tokens, needle is a string that should span
    # the tokens
    starts = np.roll(np.cumsum(list(map(len, haystack))), 1)
    starts = np.append(starts, starts[0])
    starts[0] = 0
    cat_haystack = ''.join(haystack)
    # we might get a match that doesn't correspond to a token
    begin = 0
    while begin < len(cat_haystack):
        start = cat_haystack.find(needle, begin)
        if start < 0:
            return -1, -1
        if start not in starts:
            begin = start + len(needle)
            continue
        start_idx = np.where(start == starts)[0][0]
        end_idx = start_idx + 1
        while needle not in ''.join(haystack[start_idx:end_idx]):
            end_idx += 1
            assert end_idx <= len(haystack), (start_idx, end_idx, haystack,
                                              needle)
        if needle.strip() == ''.join(haystack[start_idx:end_idx]).strip():
            return start_idx, end_idx
        begin = starts[end_idx]
    return -1, -1


class _QueryParseException(Exception):
    pass


@functools.lru_cache(maxsize=None)
def _nlp():
    return spacy.load('en_core_web_lg', disable=['parser', 'ner'])


def _parse_span(query_json):
    conds = []
    question = query_json['question']
    # wikisql-specific weirdness: everything is lowercase
    question_tokens = [x.lower() for x in question['original']]
    for _, _, literal in query_json['query']['conds']:
        cond_lit = ''.join(o.lower() + a.lower()
                           for o, a in zip(literal['gloss'], literal['after']))
        start, end = _find_sublist(question_tokens, cond_lit)
        if start < 0:
            raise _QueryParseException(
                'expected literal {} to appear in question {}'.format(
                    cond_lit, question_tokens))
        conds.append((start, end))
    return conds


def detokenize(original, after, drop_last=False):
    """
    Detokenize a list of original strings followed by their
    successive whitespace, where the last whitespace may be
    dropped.
    """
    if drop_last:
        after = after[:]
        if after:
            after[-1] = ''
    return ''.join(o + a for o, a in zip(original, after))


def _merge_vocabs(vocabs, pretrain):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], collections.Counter())
    return torchtext.vocab.Vocab(
        merged, vectors=pretrain, specials=SPECIALS, max_size=None)


# hacks until https://github.com/pytorch/text/issues/323 is resolved
# allowing for pickling the datasets' vocabulary


def _tokenize(s):
    return s.split()


def _vocab_get(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _vocab_set(self, state):
    self.__dict__.update(state)
    self.stoi = collections.defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = _vocab_get
torchtext.vocab.Vocab.__setstate__ = _vocab_set


@functools.lru_cache(maxsize=None)
def pretrained_vocab(toy):
    """fetches a small or large pretrained vocabulary"""
    # glove init magnitude based on http://anie.me/On-Torchtext/
    # TODO try 0 init for unk here
    if toy:
        pretrained = torchtext.vocab.GloVe(
            name='6B',
            cache=os.path.join(flags.FLAGS.dataroot, '.vector_cache'),
            dim=50,
            unk_init=lambda x: torch.nn.init.normal_(x, std=0.5))
    else:
        pretrained = torchtext.vocab.GloVe(
            name='840B',
            cache=os.path.join(flags.FLAGS.dataroot, '.vector_cache'),
            dim=300,
            unk_init=lambda x: torch.nn.init.normal_(x, std=0.5))
    return pretrained


def _annotate_queries(jsonl_path, toy):
    # read all json in and annotate with a spacy tagger in a single pass
    # we also use this opportunity to re-tokenize with a spacy tokenizer.
    track.debug('reading json data from {}', jsonl_path)

    track.debug('  reading json data into memory')
    with open(jsonl_path, 'r') as f:
        max_lines = 1000 if toy else _count_lines(jsonl_path)
        query_jsons = []
        for line in tqdm(itertools.islice(f, max_lines), total=max_lines):
            query_jsons.append(json.loads(line))

    nlp = _nlp()
    track.debug('  parsing json into dataset')

    words = (detokenize(q['question']['gloss'], q['question']['after'])
             for q in query_jsons)
    pipeline = nlp.pipe(words, batch_size=512, n_threads=_n_procs())
    for i, doc in enumerate(tqdm(pipeline, total=len(query_jsons))):
        ent = [tok.tag_ for tok in doc]
        tok = [_process_token(tok.lemma_, toy) for tok in doc]
        original = [tok.text_with_ws for tok in doc]
        query_jsons[i]['question'] = {}
        query_jsons[i]['question']['ent'] = ent
        query_jsons[i]['question']['tok'] = tok
        query_jsons[i]['question']['original'] = original
        yield query_jsons[i]


def _n_procs():
    try:
        return mp.cpu_count()
    except NotImplementedError:
        return 1


def _process_token(tok, toy):  # pylint: disable=too-many-return-statements
    vocab = pretrained_vocab(toy).stoi
    # TODO train these special separately, perhaps?
    # TODO or maybe give them good approximate nearby in-glove meanings
    # like SOME_NUMBER can literally be 'number' -> not the best edge case
    # but easy to impl
    if tok in vocab:
        return tok

    # we don't use textacy's preprocess_text since we have
    # different outcomes when the token matches a certain pattern
    tok = textacy.preprocess.fix_bad_unicode(tok)
    tok = textacy.preprocess.transliterate_unicode(tok)

    if tok in vocab:
        return tok

    # We consider a couple special cases to reduce simple OOV tokens
    # since the data is pretty unclean, to be honest. These are rough
    # regexes and they only check for a prefix match but they do simplify
    # the vocabulary that the model has to handle quite a bit. Note that some
    # common patterns like '1' might match these regexes but are already in
    # glove with a well-trained embedding.

    if _exact_match(textacy.constants.URL_REGEX, tok):
        return SOME_URL_WORD

    if _exact_match(textacy.constants.EMAIL_REGEX, tok):
        return SOME_EMAIL_WORD

    try:
        dateparser(tok)
        return SOME_DATE_WORD
    except (ValueError, OverflowError):
        pass

    try:
        locale.atoi(tok)
        return SOME_INT_WORD
    except ValueError:
        try:
            locale.atof(tok)
            return SOME_FLOAT_WORD
        except ValueError:
            try:
                float(tok)
                return SOME_FLOAT_WORD
            except ValueError:
                pass

    if textacy.constants.NUMBERS_REGEX.match(tok):
        return SOME_NUMPREFIX_WORD

    # consider adding hex hashes, version strings
    return tok


def _exact_match(regex, tok):
    m = regex.match(tok)
    if not m:
        return False
    if m.group(0) != tok:
        return False
    return True
