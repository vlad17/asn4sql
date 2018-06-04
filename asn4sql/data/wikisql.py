"""
This module procures data loaders for various datasets.

Note datasets are cached.
"""

import os
import itertools
import json
import re

from tqdm import tqdm
from babel.numbers import parse_decimal, NumberFormatError

from .fetch import check_or_fetch
from .. import log
from ..dbengine import DBEngine


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
    def detokenize(ls):
        """Detokenize a list of tokens"""
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
    a condition), and the Token containing the literal being
    evaluated against."""

    def __init__(self, col_idx, op_idx, condition_literal):
        self.col_idx = col_idx
        self.op_idx = op_idx
        self.condition_literal = condition_literal

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __repr__(self):
        return 'Condition({}, {}, {})'.format(self.col_idx, self.op_idx,
                                              self.condition_literal)


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

    TODO: interop with unrestricted SQL AST.

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
        self.conds = []
        for col_idx, op_idx, words in query_json['query']['conds']:
            assert op_idx >= 0 and op_idx <= 2, op_idx
            normalized_words = Token.from_dict_list(words)
            cond = Condition(col_idx, op_idx, normalized_words)
            self.conds.append(cond)
        self.question = Token.from_dict_list(query_json['question'])

        self.column_descriptions = [
            Token.from_dict_list(colname)
            for colname in query_json['table']['header']
        ]
        self.table_id = query_json['table_id']
        if not self.table_id.startswith('table'):
            self.table_id = 'table_{}'.format(self.table_id.replace('-', '_'))

        self.schema = db.get_schema(self.table_id)

        _validate(self, query_json)

    def query_and_params(self):
        """return the true parameterized query and parameter mapping"""
        # TODO: clean this method up
        select = 'col{}'.format(self.sel_idx)
        agg = Query.AGGREGATION[self.agg_idx]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}
        lower = True  # all dirty strings are lower cased for some reason?
        for cond in self.conds:
            col_index, op = cond.col_idx, cond.op_idx
            val = Token.detokenize(cond.condition_literal)
            if lower and isinstance(val, (str, bytes)):
                val = val.lower()
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

        Note these are not the real queries being run/generated.
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

    train_db, train_queries = _load_wikisql_data(
        os.path.join(wikisql_dir, 'annotated', 'train.jsonl'),
        os.path.join(wikisql_dir, 'dbs', 'train.db'), toy)
    val_db, val_queries = _load_wikisql_data(
        os.path.join(wikisql_dir, 'annotated', 'dev.jsonl'),
        os.path.join(wikisql_dir, 'dbs', 'dev.db'), toy)
    test_db, test_queries = _load_wikisql_data(
        os.path.join(wikisql_dir, 'annotated', 'test.jsonl'),
        os.path.join(wikisql_dir, 'dbs', 'test.db'), toy)

    # TODO:
    # 1. The tokens returned here are generally unclean. The query tokens
    #    across condition literals, questions, and columns can be viewed with
    #    the following recipe (sorted by length).
    #    conds = [norm_str.token for q in queries
    #             for cond in q.conds for norm_str in cond.condition_literal]
    #    question = [norm_str.token for q in queries
    #                for norm_str in q.question]
    #    columns = [norm_str.token for q in queries
    #               for c in q.columns for norm_str in c]
    #    sorted(list(set(question)), key=len)
    #    E.g., tokens like "alphabet/script" or "college/junior/club" show up,
    #    and they should be broken up into multiple tokens. Some tokens are
    #    just punctuation and that may mess stuff up too.
    #    Consider using nltk here.
    # 2. When strings are broken up, the whitespace after the string is
    #    preserved (and stored in the corresponding Token). This
    #    is not quite a disciplined approach; sometimes the unnormalized
    #    word corresponding to a token may be associated with punctuation
    #    before and after it. For example, currently the string
    #    "apple (fruit)" gets tokenized into [apple, (, fruit, )],
    #    but a much more natural organization would be [apple, fruit]; with
    #    some metadata associated with fruit to indicate it should be wrapped
    #    in parenthesis. The whitespace between words is also essential,
    #    because the column names need to be reconstructed accurately, with
    #    fully recovered spacing and hyphenation.
    # 3. Type information about the columns should be made
    #    into an enumeration and stored within the queries, as it may
    #    be useful to the algorithm (e.g., numerical condition literals should
    #    show up in numerical columns. This would need to be extracted from
    #    the DB schema.
    # 4. Inspect which tokens are not available in the glove embedding.
    # 5. See the SEMPRE paper for various preprocessing tricks to identify
    #    literals; proper nouns can be recorded as such, and numbers can
    #    be correspondingly tagged.
    # 6. It seems like we may need to get a sample of table contents or at the
    #    very least additional columnar information. For example, the columns
    #    might contain a low-cardinality set of values that could be useful
    #    in answering queries with synonyms (e.g., sports table,
    #    what was the top-scoring men's basketball team in 2008, where the
    #    sports table has league = {wnba, nba, nhl, nfl} needs to learn
    #    that the table has an "nba" entry.

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
        try:
            max_lines = 1000 if toy else _count_lines(jsonl_path)
            for line in tqdm(itertools.islice(f, max_lines), total=max_lines):
                query_json = json.loads(line)
                query = Query(query_json, db)
                queries.append(query)
        except:
            print('current query:\n{}'.format(
                json.dumps(query_json, sort_keys=True, indent=4)))
            raise

    return db, queries


def _parse_exactly_one(query_words, key):
    idx = query_words.index(key)
    assert key not in query_words[idx + 1:]
    return idx


def _extract_delimited(query_words, key, begin=0, end=-1):
    end = end if end >= 0 else len(query_words)
    if key not in query_words[begin:end]:
        return []
    start_index = query_words.index(key, begin, end)
    grouped = itertools.groupby(
        range(start_index, end), lambda x: query_words[x] == key)
    return list(list(group) for k, group in grouped if not k)


def _validate(query, query_json):
    # TODO: clean up and comment this crap
    true_query = query_json['seq_output']
    true_query_words = true_query['words']
    # generic seq_output structure is
    # symselect symagg <AGGREGATION>? symcol <token>*
    # (symwhere (symcol <token>* symop <CONDITIONAL> <token>*)
    #  (symand symcol <token>* symop <CONDITIONAL> <token>*)*)?

    # find and validate locations of main query indicators
    select_sym_idx, agg_sym_idx, end_sym_idx = (_parse_exactly_one(
        true_query_words, k) for k in ['symselect', 'symagg', 'symend'])
    where_sym_idx = -1
    if 'symwhere' in true_query_words:
        where_sym_idx = true_query_words.index('symwhere')
    assert end_sym_idx == len(true_query_words) - 1, (
        end_sym_idx, len(true_query_words) - 1)
    assert select_sym_idx == 0, select_sym_idx
    assert agg_sym_idx == 1, agg_sym_idx

    # find and validate selection columns
    col_end_idx = where_sym_idx if where_sym_idx >= 0 else end_sym_idx
    select_cols_idxs = _extract_delimited(true_query_words, 'symcol',
                                          agg_sym_idx + 1, col_end_idx)
    assert len(select_cols_idxs) == 1, \
        'need exactly 1 selection column, found {}'.format(
            select_cols_idxs)
    select_col_start_idx = select_cols_idxs[0][0]
    if agg_sym_idx + 2 < select_col_start_idx:
        agg = true_query_words[agg_sym_idx + 1]
        assert query.AGGREGATION.index(agg.upper()) == query.agg_idx, \
            'aggregation index {} does not match query aggregation {}'.format(
                query.agg_idx, agg)
        assert agg_sym_idx + 3 == select_col_start_idx, \
            'symagg index {} must be at most 3 before the first ' \
            'column name index {} with {} agg'.format(
                agg_sym_idx, select_col_start_idx, agg)
    else:
        assert query.agg_idx == 0, \
            'query with aggregation index {} right before column start ' \
            'index {} should have selection index 0'.format(
                agg_sym_idx, select_col_start_idx)
        assert agg_sym_idx + 2 == select_col_start_idx, \
            'symagg index {} must be 2 before the first ' \
            'column name index {} when null agg'.format(
                agg_sym_idx, select_col_start_idx)

    def _find_col_from_idxs(col_idxs):
        col_tokens = [true_query_words[i] for i in col_idxs]
        for i, col_strs in enumerate(query.column_descriptions):
            if [col_str.token for col_str in col_strs] == col_tokens:
                return i
        raise ValueError(
            'column with tokens {} not found -- columns are\n{}'.format(
                col_tokens, query.column_descriptions))

    query_sel_idx = _find_col_from_idxs(select_cols_idxs[0])
    assert query_sel_idx == query.sel_idx, \
        'selection column idx in query {} does not match ' \
        'that of sel_idx {}'.format(query_sel_idx, query.sel_idx)

    def _parse_where_col_idx(where_col_idx):
        has_symand = true_query_words[where_col_idx[-1]] == 'symand'
        where_col = [true_query_words[i] for i in where_col_idx]
        op_sym_idx = _parse_exactly_one(where_col, 'symop')
        op = true_query_words[where_col_idx[op_sym_idx + 1]]
        cond_sym_idx = _parse_exactly_one(where_col, 'symcond')
        assert op_sym_idx + 2 == cond_sym_idx, (op_sym_idx, cond_sym_idx)
        cond_col_idxs = where_col_idx[:op_sym_idx]
        cond_lit_idxs = where_col_idx[cond_sym_idx + 1:]
        if has_symand:
            cond_lit_idxs = cond_lit_idxs[:-1]
        col_idx = _find_col_from_idxs(cond_col_idxs)
        op_idx = query.CONDITIONAL.index(op)
        literal = Token.from_dict_list(true_query, cond_lit_idxs)
        return has_symand, Condition(col_idx, op_idx, literal)

    if where_sym_idx >= 0:
        where_cols_idxs = _extract_delimited(true_query_words, 'symcol',
                                             where_sym_idx + 1, end_sym_idx)
        assert where_cols_idxs, 'expecting at least one where column'
        has_symands, parsed_conds = zip(
            *map(_parse_where_col_idx, where_cols_idxs))
        assert all(has_symands[:-1])
        assert list(parsed_conds) == list(query.conds)
    else:
        assert query.conds == []
