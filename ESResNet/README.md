# ESResNet
## Environmental Sound Classification Based on Visual Domain Models

This repository contains implementation of the models described in the paper [arXiv:2004.07301](https://arxiv.org/abs/2004.07301) (submitted to ICPR 2020).

### How to run the model

The required Python version is >= 3.7.

#### ESResNet


##### On the [C-MDPS 1.0RPS binary](https://github.com/karolpiczak/ESC-50) dataset
    python main.py --config protocols/CMDPS/esresnet-cmdps-cv1.json --Dataset.args.root /path/to/dataset/1.0RPS

##### On the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset
    python main.py --config protocols/esc50/esresnet-esc50-cv1.json --Dataset.args.root /path/to/ESC50
