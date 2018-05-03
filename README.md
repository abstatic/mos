# Breaking the Softmax Bottleneck: A High-Rank Language Model

## Requirements

Python 3.6, PyTorch 0.2.0

## Download the data

```./get_data.sh```

## Train the models (to reproduce our results)

### Penn Treebank

First, train the model

```python main.py --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 20.0 --epoch 1000 --nhid 960 --nhidlast 620 --emsize 280 --n_experts 15  --save PTB --single_gpu```

Second, finetune the model

```python finetune.py --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 25.0 --epoch 1000 --nhid 960 --emsize 280 --n_experts 15  --save PATH_TO_FOLDER --single_gpu```

where `PATH_TO_FOLDER` is the folder created by the first step (concatenation of PTB with a timestamp).

Third, run dynamic evaluation

```python dynamiceval.py --model PATH_TO_FOLDER/finetune_model.pt --lamb 0.075```

## Acknowledgements

A large portion of this repo is borrowed from the following repos:
https://github.com/salesforce/awd-lstm-lm and https://github.com/benkrause/dynamic-evaluation


