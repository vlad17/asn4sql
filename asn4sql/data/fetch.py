"""
Simple fetch utilities.
"""

import os
import subprocess

from absl import flags
import torch

from .. import log

flags.DEFINE_string('dataroot', './data', 'data caching directory')


def check_or_fetch(directory, filename, url):
    """
    Short circuits if directory exists with the given file in the dataroot.

    Otherwise, creates the directory there and fetches the file from
    the given URL into the directory, uncompressing it in the process
    (.tgz or .zip expected as the extensions to filename).

    Always returns the full path to the directory in the dataroot.
    """
    directory = os.path.join(flags.FLAGS.dataroot, directory)
    if os.path.isfile(os.path.join(directory, filename)):
        return directory

    os.makedirs(directory, exist_ok=True)
    # can replace with torchtext.utils.download_from_url, perhaps
    subprocess.check_call(
        ['wget', '--quiet', url, '--output-document', filename], cwd=directory)
    _, ext = os.path.splitext(filename)

    if ext == '.tgz':
        subprocess.check_call(['tar', 'xzf', filename], cwd=directory)
    elif ext == '.zip':
        subprocess.check_call(['unzip', '-qq', filename], cwd=directory)
    else:
        raise ValueError('unknown extension {}'.format(ext))

    return directory


def cached_fetch(path, gen):
    """
    Checks dataroot for the path, and returns the cached pickled object if
    it exists. Else generates the object with the given gen() function
    and saves it for next time.

    Uses torch save/load.
    """
    path = os.path.join(flags.FLAGS.dataroot, path)
    if os.path.isfile(path):
        log.debug('cache hit {}', path)
        return torch.load(path)
    log.debug('cache miss {}', path)
    obj = gen()
    torch.save(obj, path)
    return obj
