# CorSW

Official research code for **A Sliced-Wasserstein Framework on Correlation
Matrices for EEG Decoding** (KDD 2026).

CorSW combines correlation-manifold self-attention with the Correlation
Sliced-Wasserstein (CorSW) regularizer for source-only domain generalization in
EEG decoding. The code supports the Off-Log Metric (OLM), Log-Scaled Metric
(LSM), and the mixed CorAtt geometry used in the paper.

## Repository contents

| Path | Purpose |
| --- | --- |
| `models/CorAtt.py` | CorAtt layers and dataset-specific networks |
| `models/corsw.py` | Metric-aware CorSW distance and domain-generalization loss |
| `utils/functions.py` | Training, evaluation, checkpointing, and reproducibility utilities |
| `CorrAttBCI.py` | BCIC-IV-2a experiment |
| `CorrAttBciCha.py` | BCI-ERN experiment |
| `CorrAttMamem.py` | MAMEM-SSVEP-II experiment |
| `conf/` | Hydra experiment configurations |

Generated data, checkpoints, Hydra outputs, and result files are intentionally
excluded from version control.

## Installation

Python 3.9 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## Data

Download each dataset from its official source and prepare the MATLAB files
expected by the loaders:

| Dataset | Default directory | Expected files and keys |
| --- | --- | --- |
| [BCIC-IV-2a](https://www.bbci.de/competition/iv/) | `data/BCICIV_2a_mat1/` | `BCIC_S##_T.mat` (`x_train`, `y_train`) and `BCIC_S##_E.mat` (`x_test`, `y_test`) |
| [MAMEM-SSVEP-II](https://doi.org/10.6084/m9.figshare.3153409.v4) | `data/MAMEM/` | `U###.mat` (`x_test`, `y_test`) |
| [BCI-ERN](https://www.kaggle.com/competitions/inria-bci-challenge/data) | `data/BCIcha/` | `Data_S##_Sess.mat` (`x_test`, `y_test`) |

The repository does not redistribute EEG data. Paths can be overridden either
with Hydra's `data_path=...` argument or by setting `CORSW_DATA` to the
parent data directory.

## Running experiments

All configurations default to CPU so that configuration and smoke checks work
on any machine. For GPU training, pass `device=cuda:0`.

```bash
# BCIC-IV-2a, subject 1
python CorrAttBCI.py subject=1 device=cuda:0

# BCI-ERN, subject 2, held-out session 1
python CorrAttBciCha.py subject=2 test_session=1 device=cuda:0

# MAMEM, subject 1, held-out session 1
python CorrAttMamem.py subject=1 test_session=1 device=cuda:0
```

Important CorSW overrides are:

```text
swd_weight=0.3
swd_num_projections=300
swd_p=2
swd_metric=olm|lsm
swd_input_space=pullback|correlation
metric=olm|lsm|mix
```

The training models currently expose their final CorAtt features in pullback
coordinates, so their default is `swd_input_space=pullback`. To compute CorSW
directly from genuine full-rank correlation matrices, use the public API:

```python
from models.corsw import corsw_distance

loss = corsw_distance(
    source_correlations,
    target_correlations,
    metric="olm",                 # or "lsm"
    input_space="correlation",   # expects full-rank correlation matrices
    num_projections=300,
)
```

Hydra multiruns can distribute subjects or folds with the Joblib launcher. For
example:

```bash
python CorrAttMamem.py --multirun subject=1,2 test_session=1,2 device=cuda:0
```

After all jobs finish, aggregate their saved run files:

```bash
python CorrAttMamem.py mode=aggregate
```

Checkpoints are written below `checkpoints/`, per-run arrays below `results/`,
and Hydra logs below `outputs/` unless overridden.

## Citation

```bibtex
@inproceedings{hu2026corsw,
  title     = {A Sliced-Wasserstein Framework on Correlation Matrices for EEG Decoding},
  author    = {Hu, Chen and Wang, Rui and Zhou, Jiale and Yi, Jingjun and Jin, Shaocheng and Song, Yidong and Zheng, Yefeng},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year      = {2026},
  doi       = {10.1145/3770855.3818864}
}
```

## Acknowledgments and license

This repository builds on the
[MAtt](https://github.com/CECNL/MAtt) and
[CorAtt](https://github.com/ChenHu-ML/CorAtt) implementations.

CorSW is released under the MIT License. See [`LICENSE`](LICENSE).
