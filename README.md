# PyTorch 3D Convolutional Neural Network
This code is for training a 3D Convolutional Neural Network on the [LUNA16](https://luna16.grand-challenge.org/home/) dataset in order to detect malignant nodules. I am hopeful that this can be used as the first step towards solving the [DSB 2017 challenge](https://www.kaggle.com/c/data-science-bowl-2017).

## Usage
### Training
```
usage: main.py [-h] [--batchSize BATCHSIZE] [--testBatchSize TESTBATCHSIZE]
               [--nEpochs NEPOCHS] [--lr LR] [--step STEP] [--cuda]
               [--resume RESUME] [--start-epoch START_EPOCH] [--clip CLIP]
               [--threads THREADS] [--momentum MOMENTUM]
               [--weight-decay WEIGHT_DECAY] [--pretrained PRETRAINED]

PyTorch Luna_X-Net

optional arguments:
  -h, --help            show this help message and exit
  --batchSize BATCHSIZE
                        training batch size
  --testBatchSize TESTBATCHSIZE
                        testing batch size
  --nEpochs NEPOCHS     number of epochs to train for
  --lr LR               Learning Rate. Default=0.001
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs, Default: n=15
  --cuda                use cuda?
  --resume RESUME       path to latest checkpoint (default: none)
  --start-epoch START_EPOCH
                        manual epoch number (useful on restarts)
  --clip CLIP           Clipping Gradients. Default=10
  --threads THREADS     number of threads for data loader to use
  --momentum MOMENTUM   momentum
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        weight decay, Default: 0
  --pretrained PRETRAINED
                        path to pretrained model (default: none)


```

### Todo

Network Optimization