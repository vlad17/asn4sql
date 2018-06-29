# Abstract Syntax Networks for SQL Code Synthesis [![Build Status](https://travis-ci.com/vlad17/asn4sql.svg?token=xAqzxKFpxN3pG4om3z4n&branch=master)](https://travis-ci.com/vlad17/asn4sql)

## Setup

See `setup.py` for necessary python packages. Requires a linux x64 box. Note the use of a `spacy` entity tagger.

```
conda create -y -n asn4sql-env python=3.5
source activate asn4sql-env
./scripts/install-pytorch.sh
pip install --upgrade git+https://github.com/richardliaw/track.git@master#egg=track
pip install --no-cache-dir --editable .
python -m spacy download en_core_web_lg
```

## Scripts

All scripts are available in `scripts/`, and should be run from the repo root in the `asn4sql-env`.

| script | purpose |
| ------ | ------- |
| `lint.sh` | invokes `pylint` with the appropriate flags for this repo |
| `tests.sh` | runs tests |
| `install-pytorch.sh` | infer python and cuda versions, use them to install pytorch |
| `format.sh` | auto-format the entire `asn4sql` directory |

## Example

All mainfiles are documented. Run `python asn4sql/main/*.py --help` for any `*` for details.

```
TODO example code
```
