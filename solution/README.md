# Possession solution

Rule-based possession labelling for the BAR-ATH match. It uses no trained model and no labels.
Method, rules and limitations are in [inka-analytics.md](inka-analytics.md).

## Install

Python 3.11 is required. Tested with 3.11.3 on Windows 11.

```bash
pip install -r solution/requirements.txt
```

## Data

The match data is not in this repository. Unpack the bundle into `data/` as described in
[DATA.md](../DATA.md), or set `MTAH_DATA_DIR` to its location.

The submission file needs only three tactical assets: the tracking `.txt`, its `.xml` metadata and
the Smart Tagging `.xml`. The broadcast tracking is needed only by the comparison scripts.

## Regenerate the submission

Run these from the repository root. The method uses no randomness.

```bash
python solution/possession_v4.py --out evaluation/results/submission.csv
```

```bash
python evaluation/validate_submission.py evaluation/results/submission.csv
```

The expected SHA-256 is in [inka-analytics.md](inka-analytics.md#reproduction). The script
writes `\n` line endings on every operating system, so line endings cannot change the hash.

## Files

| file | purpose |
|---|---|
| `possession_v4.py` | entry point of the submitted version |
| `possession_v1.py` | tracking reader, control evidence and dead-ball detection |
| `possession_v2.py` | score per frame and Viterbi decoding |
| `possession_v3.py` | ball speed derived from positions, for the control evidence |
| `broadcast_ball.py` | derived speed; broadcast sync, used only for checks |
| `compare_versions.py` | compares versions on two label-free exams |
| `exam_handover.py` | blind exam on possession changes with simulated ball gaps |

`possession_v1.py` to `v3.py` also run on their own, which reproduces each earlier version.
