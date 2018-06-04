"""Datasets used for SQL synthesis"""

from absl import flags

from .wikisql import wikisql

flags.DEFINE_string('dataroot', './data', 'data caching directory')
