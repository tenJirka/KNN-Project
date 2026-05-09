# Car re-identification

## Authors

- Jiří Stehlík, xstehl23@vutbr.cz
- Filip Černý, xcerny81@vutbr.cz
- Marek Halamka, xhalam16@vutbr.cz

## Usage

### Dependency installation

Project dependencies are managed by `poetry`. To install all dependencies run:

```bash
petry install
```

> [!NOTE]
> `Poetry` is unable to install correct PyTorch version for you. If you want to install specific [PyTorch](https://pytorch.org/get-started/locally/) version manually (as system package for example), you can install only other dependencies (without `torch`) via `poetry install --without system-deps`.

### Training

Training scripts are in `train/`.

Usage:

```bash
python train.py <checkpoints_path> <model_name/checkpoint_path> [--fit/--all]
```

Will train `model_name` on `VeRi` dataset for at least 15 epochs with `batch_size=200`.

Then it will be stopped by early stopping based on `mAP`, `min_delta=0.003` and `patience=5`.

Top 3 checkpoints will be saved to `checkpoints_path`.

By default, model is trained only on VeRi dataset. If `--fit` is specified, training is done on FIT dataset only and if `--all`, training is done on both.

### Testing

Testing scripts are in `train/`.

Usage:

```bash
python test.py <checkpoint_path>
```

This will evaluate model on both VeRi and FIT datasets.

## Datasets

For model training, you have to obtain and your own copy for datasets. To get more information, look into [datasets](datasets/README.md) folder.
