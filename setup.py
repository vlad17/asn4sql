"""setup.py for asn4sql"""

from setuptools import setup

install_requires = [
    'torchtext>=0.2.3',
    'torchvision>=0.1.8',
    'numpy>=0.14',
    'matplotlib>=2.2.2',
    'absl-py>=0.1.13',
    'scikit-learn>=0.19.1',
    'pylint>=1.9.1',
    'yapf>=0.22.0'
]

setup(name="asn4sql", author="RISE Lab", install_requires=install_requires)
