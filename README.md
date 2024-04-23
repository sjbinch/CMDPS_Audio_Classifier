# ESResNet
## Environmental Sound Classification Based on Visual Domain Models

This repository contains implementation of the models described in the paper [arXiv:2004.07301](https://arxiv.org/abs/2004.07301) (submitted to ICPR 2020).

### How to run the model

The required Python version is >= 3.7.

#### ESResNet


##### On the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset
    python main.py --config protocols/esc50/esresnet-esc50-cv1.json --Dataset.args.root /path/to/ESC50

##### On the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset
    python main.py --config protocols/esc50/esresnet-esc50-cv1.json --Dataset.args.root /path/to/ESC50



#### Reproduced results

##### [LMCNet](https://www.mdpi.com/1424-8220/19/7/1733/pdf) on the [UrbanSound8K](https://urbansounddataset.weebly.com/) dataset
    python main.py --config protocols/us8k/lmcnet-us8k-cv1.json --Dataset.args.root /path/to/UrbanSound8K
