# The Effects of Overparameterization on Sharpness-aware Minimization: An Empirical and Theoretical Analysis

This repository contains JAX/Flax source code for arXiv paper [The Effects of Overparameterization on Sharpness-aware Minimization: An Empirical and Theoretical Analysis](https://arxiv.org/abs/2311.17539) by [Sungbin Shin](https://ssbin4.github.io/)<sup>&ast;</sup>, [Dongyeop Lee](https://edong6768.github.io/)<sup>&ast;</sup>, [Maksym Andriushchenko](https://www.andriushchenko.me/), and [Namhoon Lee](https://namhoonlee.github.io/).

## TL;DR

We analyze the effects of overparameterization on several theoretical and empirical aspects of sharpness-aware minimization.

## Abstract

Training an overparameterized neural network can yield minimizers of the same level of training loss and yet different generalization capabilities. With evidence that indicates a correlation between sharpness of minima and their generalization errors, increasing efforts have been made to develop an optimization method to explicitly find flat minima as more generalizable solutions. This sharpness-aware minimization (SAM) strategy, however, has not been studied much yet as to how overparameterization can actually affect its behavior. In this work, we analyze SAM under varying degrees of overparameterization and present both empirical and theoretical results that suggest a critical influence of overparameterization on SAM. Specifically, we first use standard techniques in optimization to prove that SAM can achieve a linear convergence rate under overparameterization in a stochastic setting. We also show that the linearly stable minima found by SAM are indeed flatter and have more uniformly distributed Hessian moments compared to those of SGD. These results are corroborated with our experiments that reveal a consistent trend that the generalization improvement made by SAM continues to increase as the model becomes more overparameterized. We further present that sparsity can open up an avenue for effective overparameterization in practice.

![fig](./figures/main.png)

## Environments

### Python
- python 3.8.0

### cuda
- cuda 11.4.4
- cudnn 8.6.0
- nccl 2.11.4

### Dependencies
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py --workdir={logging_dir} --config={config_file}
```

Examples of the config files are located in the `configs` directory.

The degree of overparameterization is determined by `config.num_neurons` for MLP and `config.num_filters` for ResNet, while the degree of sparsification is determined by `config.sparsity`.

## Citation
```
@article{shin2023effects,
  title={The Effects of Overparameterization on Sharpness-aware Minimization: An Empirical and Theoretical Analysis},
  author={Shin, Sungbin and Lee, Dongyeop and Andriushchenko, Maksym and Lee, Namhoon},
  journal={arXiv preprint arXiv:2311.17539},
  year={2023}
}
```
