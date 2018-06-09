"""
DBEngine taken from SQLNet, which in turn took the DBEngine from the original
WikiSQL repository.
"""

import os
import re
import functools

from babel.numbers import parse_decimal, NumberFormatError
import records

from . import data

_SCHEMA_RE = re.compile(r'\((.+)\)')


class DBEngine:
    """SQLite interface to actual record information"""

    def __init__(self, fdb):
        self.db = records.Database('sqlite:///{}'.format(fdb))
        self.db_path = os.path.abspath(fdb)

    def execute_query(self, query):
        """executes the query on the given DB"""
        # assumes there's only one selection column
        query_str, query_param_map = self.query_and_params(query)
        query_param_map = {k: v for k, (col, v) in query_param_map.items()}
        out = self.db.query(query_str, **query_param_map)
        return [next(iter(o.values())) for o in out]

    @functools.lru_cache(maxsize=None)
    def get_schema(self, table_id):
        """Return the column name to type mapping"""
        table_info = self.db.query(
            'SELECT sql from sqlite_master WHERE tbl_name = :name',
            name=table_id).all()[0].sql.replace('\n', '')
        schema_str = _SCHEMA_RE.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(', '):
            c, t = tup.split()
            schema[c] = t
        return schema

    def query_and_params(self, ex):
        """
        Return the true parameterized query and parameter mapping of a query,
        given a torchtext Example of the WikiSQL data.
        """
        select = 'col{}'.format(ex.sel)
        agg = data.wikisql.AGGREGATION[ex.agg]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}

        schema = self.get_schema(ex.table_id)
        param = 0

        for col_idx, op, l, r in zip(ex.cond_col, ex.cond_op, ex.cond_span_l,
                                     ex.cond_span_r):
            val = data.wikisql.detokenize(
                ex.original[l:r], ex.after[l:r], drop_last=True)
            val = val.lower()  # weird wikisql standard is to lowercase
            if schema['col{}'.format(col_idx)] == 'real' and not isinstance(
                    val, (int, float)):
                try:
                    val = float(parse_decimal(val))
                except NumberFormatError:
                    _NUM_RE = re.compile(r'[-+]?\d*\.\d+|\d+')
                    val = float(_NUM_RE.findall(val)[0])
            where_clause.append('col{} {} :param{}'.format(
                col_idx, data.wikisql.CONDITIONAL[op], param))
            where_map['param{}'.format(param)] = ('col{}'.format(col_idx), val)
            param += 1
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)
        query_str = 'SELECT {} FROM {} {}'.format(select, ex.table_id,
                                                  where_str)
        return query_str, where_map

    def interpolated_query(self, ex):
        """
        Fake param-interpolated query; for debugging purposes only.
        Assumes that strings have no quotes and column names only need
        to be quoted if they have whitespace. To make the query look prettier
        we get rid of the clean col<i> names and replace them with the
        column descriptions.

        Note these are not the real queries being run/generated; those
        use query parameterization and the mapped column names col<i>.

        Also note the column names are reconstructed from tokens rather than
        the original strings so the spacing may be off.
        """
        query_str, param_map = self.query_and_params(ex)
        schema = self.get_schema(ex.table_id)
        for k, (col, v) in param_map.items():
            if schema[col] == 'text':
                v = "'{}'".format(v)
            query_str = query_str.replace(':' + k, v)
        cols = ' '.join(ex.tbl).split(data.wikisql.SPLIT_WORD)
        if cols and not cols[-1]: # remove trailing split
            cols.pop()
        for i, col_desc in enumerate(s.strip() for s in cols):
            colname = col_desc
            if re.search(r'\s|/|\p', colname):
                colname = '"{}"'.format(colname)
            query_str = query_str.replace('col{}'.format(i), colname)
        return query_str
