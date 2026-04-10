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
> `Poetry` is unable to install correct PyTorch version for you. If you want to install specific [PyTorch](https://pytorch.org/get-started/locally/) version manually (as system package for example), you can install only other dependecies (without `torch`) via `poetry install --without system-deps`.

### Datasets

For model training, you have to obtain and your own copy for datasets. To get more information, look into [datasets](datasets/README.md) folder.
