"""
DBEngine taken from SQLNet, which in turn took the DBEngine from the original
WikiSQL repository.
"""

import re
import functools

import records

_SCHEMA_RE = re.compile(r'\((.+)\)')


class DBEngine:
    """SQLite interface to actual record information"""

    def __init__(self, fdb):
        print(fdb)
        self.db = records.Database('sqlite:///{}'.format(fdb))

    def execute_query(self, query):
        """executes the query on the given DB"""
        # assumes there's only one selection column
        query_str, query_param_map = query.query_and_params()
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
