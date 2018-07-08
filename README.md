# Abstract Syntax Networks for SQL Code Synthesis [![Build Status](https://travis-ci.com/vlad17/asn4sql.svg?token=xAqzxKFpxN3pG4om3z4n&branch=master)](https://travis-ci.com/vlad17/asn4sql)

## Setup

See `setup.py` for necessary python packages. Requires a linux x64 box. Note the use of a `spacy` entity tagger.

```
conda create -y -n asn4sql-env python=3.5
source activate asn4sql-env
./scripts/install-pytorch.sh
pip install --no-cache-dir --upgrade git+https://github.com/richardliaw/track.git@master#egg=track
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
| `launch-mps.sh` | setup up a GPU-specific MPS server (must be sourced, log path hard-coded) |
| `disable-mps.sh` | tear down up a GPU-specific MPS server (log path hard-coded) |

## Example

All mainfiles are documented. Run `python asn4sql/main/*.py --help` for any `*` for details.

```
python asn4sql/main/preprocess_data.py
export CUDA_VISIBLE_DEVICES=0
source ./scripts/launch-mps.sh
# defaults are performant accuracy-wise
# adjust --batch_size and --workers accodring to your GPU memory usage
python asn4sql/main/wikisql_specific.py --experiment_comment "my-favorite-trial"
trial=$(python -m track.trials trial_id experiment_comment=my-favorite-trial | head -2 | tail -1 | cut -d" " -f1)
python asn4sql/main/test_wikisql.py --trial $trial
./scripts/disable-mps.sh
```

## Q&A

**What do [Abstract Syntax Networks](https://arxiv.org/abs/1704.07535) have to do with this?** Nothing. This project was initially supposed to compare how much we lose in terms of accuracy by going from an incredibly problem-specific architecture for "synthesis" to a generic synthesis net, but I never got around to the latter. Besides, ASN is not the end-all of generic synthesis network architectures, see for instance [this graph-based one](https://arxiv.org/abs/1805.08490).

**Why don't you use packed sequences for seq2seq training?** I anticipate having dynamic computation graph shapes (i.e., even more dynamic than variable-length sequences). So it doesn't make sense to have a training script specialized to seq2seq-style training, when Nvidia MPS and single-GPU multiplexing can achieve good performance.
