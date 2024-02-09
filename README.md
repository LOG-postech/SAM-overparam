# Analyzing Sharpness-aware Minimization under Overparameterization

This repository contains JAX/Flax source code for arXiv paper [Analyzing Sharpness-aware Minimization under Overparameterization](https://arxiv.org/abs/2311.17539) by [Sungbin Shin](https://ssbin4.github.io/)<sup>&ast;</sup>, [Dongyeop Lee](https://edong6768.github.io/)<sup>&ast;</sup>, [Maksym Andriushchenko](https://www.andriushchenko.me/), and [Namhoon Lee](https://namhoonlee.github.io/).

## TL;DR

We analyze the effects of overparameterization on several theoretical and empirical aspects of sharpness-aware minimization.

## Abstract

Training an overparameterized neural network can yield minimizers of different generalization capabilities despite the same level of training loss. With evidence that suggests a correlation between sharpness of minima and their generalization errors, increasing efforts have been made to develop an optimization method to explicitly find flat minima as more generalizable solutions. However, this sharpness-aware minimization (SAM) strategy has not been studied much yet as to whether and how it is affected by overparameterization.

In this work, we analyze SAM under overparameterization of varying degrees and present both empirical and theoretical results that indicate a critical influence of overparameterization on SAM. Specifically, we conduct extensive numerical experiments across various domains, and show that there exists a consistent trend that SAM continues to benefit from increasing overparameterization. We also discover compelling cases where the effect of overparameterization is more pronounced or even diminished along with a series of ablation studies. On the theoretical side, we use standard techniques in optimization and prove that SAM can achieve a linear rate of convergence under overparameterization in a stochastic setting. We also show that overparameterization can improve generalization of SAM based on an analysis of two-layer networks, and further, that the linearly stable minima found by SAM have more uniform Hessian moments compared to SGD.

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
@article{shin2024analyzing,
  title={Analyzing Sharpness-aware Minimization under Overparameterization},
  author={Shin, Sungbin and Lee, Dongyeop and Andriushchenko, Maksym and Lee, Namhoon},
  journal={arXiv preprint arXiv:2311.17539},
  year={2024}
}
```
