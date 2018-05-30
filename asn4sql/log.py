"""
Imagine you want to log stuff in multiple files of a Python project. You'd
probably write some code that looks like this:

    logging.debug("Hello, World!")

Pretty simple. Unfortunately, some packages like gym seem to clobber the
default logger that is used by logging.debug. No big deal, just use a different
logger:

    logging.getLogger("mylogger").debug("Hello, World!")

Two problems stick out though. First, "mylogger" is a bit of a magic constant.
Second, that's a lot of typing for a print statement! This file makes it a
tinier bit more convenient to log stuff. In one file, run this:

    import log
    log.init(logdir)

And then from any other file, run something this:

    from log import debug
    debug("Iteration {} of {}", 1, num_iters)

The logging directory logdir is placed inside the logging root; the
invocation flags and various configuration info is placed inside.
"""

import inspect
import subprocess
import os
import shutil
import shlex
import json
import sys
import logging

from absl import flags

flags.DEFINE_boolean('quiet', False, 'suppress debug logging output to stdout')
flags.DEFINE_string('logroot', './logs', 'log root directory')

_PACKAGE_NAME = 'asn4sql'


class _StackCrawlingFormatter(logging.Formatter):
    """
    If we configure a python logger with the format string
    "%(pathname):%(lineno): %(message)", messages logged via `log.debug` will
    be prefixed with the path name and line number of the code that called
    `log.debug`. Unfortunately, when a `log.debug` call is wrapped in a helper
    function (e.g. debug below), the path name and line number is always that
    of the helper function, not the function which called the helper function.

    A _StackCrawlingFormatter is a hack to log a different pathname and line
    number. Simply set the `pathname` and `lineno` attributes of the formatter
    before you call `log.debug`. See `debug` below for an example.
    """

    def __init__(self, format_str):
        super().__init__(format_str)
        self.pathname = None
        self.lineno = None

    def format(self, record):
        s = super().format(record)
        if self.pathname is not None:
            s = s.replace('{pathname}', self.pathname)
        if self.lineno is not None:
            s = s.replace('{lineno}', str(self.lineno))
        return s


_LOGGER = logging.getLogger(_PACKAGE_NAME)
_FORMAT_STRING = "[%(asctime)-15s {pathname}:{lineno}] %(message)s"
_FORMATTER = _StackCrawlingFormatter(_FORMAT_STRING)
_LOGDIR = None


def logging_directory():
    """Return the current logging directory"""
    return _LOGDIR


def init(logdir=None):
    """Initialize the logger."""
    if logdir is not None:
        global _LOGDIR  # pylint: disable=global-statement
        logdir = os.path.join(flags.FLAGS.logroot, logdir)
        logdir = _make_log_directory(logdir)
        _LOGDIR = logdir

    handler = logging.StreamHandler()
    handler.setFormatter(_FORMATTER)
    _LOGGER.propagate = False
    _LOGGER.addHandler(handler)
    if logdir is not None:
        _LOGGER.addHandler(
            logging.FileHandler(os.path.join(logdir, 'log.txt')))
    if not flags.FLAGS.quiet:
        _LOGGER.setLevel(logging.DEBUG)
    if logdir is not None:
        debug('log directory is {}', logdir)
        _save_git_hash(os.path.join(logdir, 'githash.txt'))
        _save_invocation(os.path.join(logdir, 'invocation.txt'))
        _save_flags(os.path.join(logdir, 'flags.json'))
        flags.FLAGS.append_flags_into_file(os.path.join(logdir, 'flags.flags'))


def debug(s, *args):
    """debug(s, x1, ..., xn) logs s.format(x1, ..., xn)."""
    # Get the path name and line number of the function which called us.
    previous_frame = inspect.currentframe().f_back
    try:
        pathname, lineno, _, _, _ = inspect.getframeinfo(previous_frame)
        # if path is in cwd, simplify it
        cwd = os.path.abspath(os.getcwd())
        pathname = os.path.abspath(pathname)
        if os.path.commonprefix([cwd, pathname]) == cwd:
            pathname = os.path.relpath(pathname, cwd)
    except Exception:  # pylint: disable=broad-except
        pathname = '<UNKNOWN-FILE>.py'
        lineno = 0
    _FORMATTER.pathname = pathname
    _FORMATTER.lineno = lineno
    _LOGGER.debug(s.format(*args))


def _make_log_directory(name):
    """
    _make_log_directory(name) (where name is a path) will create the
    corresponding directory/directories.

    If the directory already exists, then it will be
    renamed name-i where i is the smallest integer such that data/name-i
    does not already exist. For example, imagine the data/ directory has the
    following contents:

        data/foo
        data/foo-old-0
        data/foo-old-1
        data/foo-old-2
        data/foo-old-3

    Then, make_data_directory("data/foo") will rename data/foo to
    data/foo-old-4 and then create a fresh data/foo directory.
    """
    name = os.path.normpath(name)
    directory = os.path.dirname(name)
    if directory:
        os.makedirs(directory, exist_ok=True)

    ctr = 0
    logdir = name
    while os.path.exists(logdir):
        logdir = name + '-old-{}'.format(ctr)
        ctr += 1
    if ctr > 0:
        print(
            'log directory {} already exists, moved old one to {}'.format(
                name, logdir),
            file=sys.stderr)
        shutil.move(name, logdir)

    os.makedirs(name)
    return os.path.abspath(name)


def _save_git_hash(filename):
    try:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(module_dir)
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=package_dir)
        # git_hash is a byte string; we want a string.
        git_hash = git_hash.decode('utf-8')
        # git_hash also comes with an extra \n at the end, which we remove.
        git_hash = git_hash.strip()
    except subprocess.CalledProcessError:
        git_hash = '<no git hash available>'
    with open(filename, 'w') as f:
        print(git_hash, file=f)


def _save_invocation(filename):
    cmdargs = [sys.executable] + sys.argv[:]
    invocation = ' '.join(shlex.quote(s) for s in cmdargs)
    with open(filename, 'w') as f:
        print(invocation, file=f)


def _save_flags(filename):
    flags_dict = flags.FLAGS.flags_by_module_dict()
    flags_dict = {
        module_name: {flag.name: flag.value
                      for flag in module_flags}
        for module_name, module_flags in flags_dict.items()
    }
    with open(filename, 'w') as f:
        json.dump(flags_dict, f, sort_keys=True, indent=4)
