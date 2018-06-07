"""
This module procures data loaders for various datasets.

Note datasets are cached.
"""

import os
import copy
import itertools
import json
import re

from tqdm import tqdm
from babel.numbers import parse_decimal, NumberFormatError

from .fetch import check_or_fetch
from .. import log
from ..dbengine import DBEngine


class _QueryParseException(Exception):
    pass


class Token:
    """WikiSQL has several forms of the same string as preprocessing.

    This stores the original string and the whitespace or punctuation
    that it has before or after it
    and the cleaned, lemmatized, lowercased version.
    """

    def __init__(self, word, gloss, after):
        self.token = word
        self.original = gloss
        self.after = after

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.token == other.token
        return False

    def __repr__(self):
        return 'Token({}, {}, {})'.format(self.token, self.original,
                                          self.after)

    @staticmethod
    def detokenize(ls, drop_last=False):
        """Detokenize a list of tokens"""
        if drop_last:
            ls = ls[:]
            if ls:
                ls[-1] = Token(ls[-1].token, ls[-1].original, '')
        return ''.join(s.original + s.after for s in ls)

    @classmethod
    def from_dict_list(cls, dict_list, subindex=None):
        """Converts a dictionary of word, gloss, after lists to a
        single list of tokens."""
        if subindex is None:
            subindex = range(len(dict_list['words']))
        sublists = (list(dict_list[key][i] for i in subindex)
                    for key in ['words', 'gloss', 'after'])
        return [cls(*args) for args in zip(*sublists)]


class Condition:
    """A condition goes after a WHERE predicate.

    In WikiSQL, all conditions are of the form <column> <op> <literal>,
    e.g., person_name=sam.

    We store the operation index into Query.CONDITIONAL, the column index
    (corresponding to a table schema referenced by the query containing
    a condition), and the literal is always assumed to be a span of the
    input question"""

    def __init__(self, col_idx, op_idx, lit_begin, lit_end):
        self.col_idx = col_idx
        self.op_idx = op_idx
        self.lit_begin = lit_begin
        self.lit_end = lit_end

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __repr__(self):
        return 'Condition(col_idx={}, op_idx={}, lit_begin={}, lit_end={})' \
            .format(self.col_idx, self.op_idx, self.lit_begin, self.lit_end)

    def literal_toks(self, question):
        """returns the literals from a question"""
        return question[self.lit_begin:self.lit_end]


class Query:
    """
    A single labelled point in the WikiSQL supervised task.

    Identifiers are stored both in a tokenized format (as a list
    of lists, each sublist containing the tokens) and the original
    format (as a list of identifiers, which may contain spaces).

    In addition to the natural question string, this contains
    table and schema information.

    Note that WikiSQL has a very simplified version of SQL, so its
    predicates all have a (potentially null) aggregation,
    single selection column, and possibly multiple binary
    conditional statements (all in conjunction).

    For this reason, Query supports a hard-coded sketch-based
    API.

    Members:

    * agg_idx - index within Query.AGGREGATION for query aggregation
    * sel_idx - single column being selected for the aggregation (which
                may be a trivial null aggregation)
    * conds - list of Condition objects; if nonnull indicates a conjunction
              of binary clauses after a WHERE.
    * question - list of tokens for the  natural language question
    * column_descriptions - list of lists of tokens for each column name,
                            offering a description of the i-th column, which
                            then actually has name col<i>
    * schema - mapping from i-th column name (e.g., col0) to type
    * table_id - string table id for query
    """

    AGGREGATION = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    CONDITIONAL = ['=', '>', '<']  # OP conditional is never in the data...

    def __init__(self, query_json, db):
        self.agg_idx = query_json['query']['agg']
        self.sel_idx = query_json['query']['sel']
        self.question = Token.from_dict_list(query_json['question'])
        question_tokens = [tok.token for tok in self.question]
        self.conds = []
        for col_idx, op_idx, words in query_json['query']['conds']:
            assert op_idx >= 0 and op_idx <= 2, op_idx
            normalized_words = Token.from_dict_list(words)
            cond_lit_toks = [tok.token for tok in normalized_words]
            start, end = _find_sublist(question_tokens, cond_lit_toks)
            if start < 0:
                raise _QueryParseException(
                    'expected literal tokens {} to appear in question {}'
                    .format(cond_lit_toks, question_tokens))
            cond = Condition(col_idx, op_idx, start, end)
            # weird wikisql standard is to lowercase query literals
            cond_lit_str = Token.detokenize(normalized_words)
            cond_lit_str = cond_lit_str.lower()
            reconstruct_lit_str = Token.detokenize(
                self.question[start:end], drop_last=True)
            reconstruct_lit_str = reconstruct_lit_str.lower()
            if reconstruct_lit_str != cond_lit_str:
                raise _QueryParseException(
                    'expected query literal "{}" to be the same when '
                    'reconstructed from the question "{}"'.format(
                        cond_lit_str, reconstruct_lit_str))
            self.conds.append(cond)

        self.column_descriptions = [
            Token.from_dict_list(colname)
            for colname in query_json['table']['header']
        ]
        self.table_id = query_json['table_id']
        if not self.table_id.startswith('table'):
            self.table_id = 'table_{}'.format(self.table_id.replace('-', '_'))

        self.schema = db.get_schema(self.table_id)
        self.db = db

    def with_prediction(self, agg_idx, sel_idx, conds):
        """
        generate the query associated with possibly different
        predicted aggregation, selection, and conditional indices.
        """
        shallow_copy = copy.copy(self)
        shallow_copy.agg_idx = agg_idx
        shallow_copy.sel_idx = sel_idx
        shallow_copy.conds = conds
        return shallow_copy

    def query_and_params(self):
        """
        return the true parameterized query and parameter mapping of this query
        """
        select = 'col{}'.format(self.sel_idx)
        agg = Query.AGGREGATION[self.agg_idx]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}

        for cond in self.conds:
            col_index, op = cond.col_idx, cond.op_idx
            val = Token.detokenize(
                cond.literal_toks(self.question), drop_last=True)
            val = val.lower()  # weird wikisql standard is to lowercase
            if self.schema['col{}'.format(
                    col_index)] == 'real' and not isinstance(
                        val, (int, float)):
                try:
                    val = float(parse_decimal(val))
                except NumberFormatError:
                    _NUM_RE = re.compile(r'[-+]?\d*\.\d+|\d+')
                    val = float(_NUM_RE.findall(val)[0])
            where_clause.append('col{} {} :col{}'.format(
                col_index, Query.CONDITIONAL[op], col_index))
            where_map['col{}'.format(col_index)] = val
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)
        query_str = 'SELECT {} FROM {} {}'.format(select, self.table_id,
                                                  where_str)
        return query_str, where_map

    def interpolated_query(self):
        """
        Fake param-interpolated query; for debugging purposes only.
        Assumes that strings have no quotes and column names only need
        to be quoted if they have whitespace. To make the query look prettier
        we get rid of the clean col<i> names and replace them with the
        column descriptions.

        Note these are not the real queries being run/generated; those
        use query parameterization and the mapped column names col<i>.
        """
        query_str, param_map = self.query_and_params()
        for k, v in param_map.items():
            if self.schema[k] == 'text':
                v = "'{}'".format(v)
            query_str = query_str.replace(':' + k, v)
        for i, col_desc in enumerate(self.column_descriptions):
            colname = Token.detokenize(col_desc)
            if re.search(r'\s|/|\p', colname):
                colname = '"{}"'.format(colname)
            query_str = query_str.replace('col{}'.format(i), colname)
        return query_str


_URL = 'https://drive.google.com/uc?id=1uMG4cjaQGNU_HRW-2TPB5g904FITKuKE'


def wikisql(toy):
    """
    Loads the WikiSQL dataset. Per the original SQLNet implementation,
    a subsampled version is returned if the toy argument is true.

    TODO what is returned
    """
    # compressed WikiSQL file after the annotation recommended in
    # https://github.com/salesforce/WikiSQL has been run.
    wikisql_dir = check_or_fetch('wikisql', 'wikisql.tgz', _URL)
    wikisql_dir = os.path.join(wikisql_dir, 'wikisql')

    train_db, train_queries = _load_wikisql_data(
        os.path.join(wikisql_dir, 'annotated', 'train.jsonl'),
        os.path.join(wikisql_dir, 'dbs', 'train.db'), toy)
    val_db, val_queries = _load_wikisql_data(
        os.path.join(wikisql_dir, 'annotated', 'dev.jsonl'),
        os.path.join(wikisql_dir, 'dbs', 'dev.db'), toy)
    test_db, test_queries = _load_wikisql_data(
        os.path.join(wikisql_dir, 'annotated', 'test.jsonl'),
        os.path.join(wikisql_dir, 'dbs', 'test.db'), toy)

    return train_db, train_queries, val_db, val_queries, test_db, test_queries


def _count_lines(fname):
    with open(fname, 'r') as f:
        return sum(1 for line in f)


def _load_wikisql_data(jsonl_path, db_path, toy):
    queries = []
    db = DBEngine(db_path)

    log.debug('reading sql data from {}', jsonl_path)
    with open(jsonl_path, 'r') as f:
        query_json = ''
        max_lines = 1000 if toy else _count_lines(jsonl_path)
        excs = []
        for line in tqdm(itertools.islice(f, max_lines), total=max_lines):
            try:
                query_json = json.loads(line)
                query = Query(query_json, db)
                queries.append(query)
            except _QueryParseException as e:
                excs.append(e.args[0])
    log.debug('dropped {} of {} queries{}{}', len(excs),
              len(excs) + len(queries), ':\n    '
              if excs else '', '\n    '.join(excs))

    return db, queries


def _find_sublist(haystack, needle):
    # my algs prof can bite me
    start, end = -1, -1
    for i in range(len(haystack)):
        end = i + len(needle)
        if needle == haystack[i:end]:
            start = i
            break
    return start, end
