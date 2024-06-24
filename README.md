# Critical Influence of Overparameterization on Sharpness-aware Minimization

This repository contains JAX/Flax source code for arXiv paper [Critical Influence of Overparameterization on Sharpness-aware Minimization](https://arxiv.org/abs/2311.17539) by [Sungbin Shin](https://ssbin4.github.io/)<sup>&ast;</sup>, [Dongyeop Lee](https://edong6768.github.io/)<sup>&ast;</sup>, [Maksym Andriushchenko](https://www.andriushchenko.me/), and [Namhoon Lee](https://namhoonlee.github.io/).

## TL;DR

We uncover both empirical and theoretical results that indicate a critical influence of overparameterization on SAM.

## Abstract

Training an overparameterized neural network can yield minimizers of different generalization capabilities despite the same level of training loss. Meanwhile, with evidence that suggests a strong correlation between the sharpness of minima and their generalization errors, increasing efforts have been made to develop optimization methods to explicitly find flat minima as more generalizable solutions. Despite its contemporary relevance to overparameterization, however, this sharpness-aware minimization (SAM) strategy has not been studied much yet as to exactly how it is affected by overparameterization. Hence, in this work, we analyze SAM under overparameterization of varying degrees and present both empirical and theoretical results that indicate a critical influence of overparameterization on SAM. At first, we conduct extensive numerical experiments across vision, language, graph, and reinforcement learning domains and show that SAM consistently improves with overparameterization. Next, we attribute this phenomenon to the interplay between the enlarged solution space and increased implicit bias from overparameterization. Further, we prove multiple theoretical benefits of overparameterization for SAM to attain (i) minima with more uniform Hessian moments compared to SGD, (ii) much faster convergence at a linear rate, and (iii) lower test error for two-layer networks. Last but not least, we discover that the effect of overparameterization is more significantly pronounced in practical settings of label noise and sparsity, and yet, sufficient regularization is necessary.

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
@article{shin2024critical,
  title={Critical Influence of Overparameterization on Sharpness-aware Minimization},
  author={Shin, Sungbin and Lee, Dongyeop and Andriushchenko, Maksym and Lee, Namhoon},
  journal={arXiv preprint arXiv:2311.17539},
  year={2024}
}
```
