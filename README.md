## Map Super Resolution
### ToDo
- [x] Add CARN method
- [x] Add different optimization method
- [x] Log the checkppoints and _logs
- [x] Log the result
- [ ] Calculating FLOPs with -> https://github.com/vra/flopth

### Optimizer
- [x] ADAM
- [x] AdamSparse
- [x] Adamax
- [x] Adadelta
      - Adagrad
- [x] ASGD
      - [x] LARS
      - [x] LAMB
- [x] RProp
      - [x] SGD
      - [x] RMSprop

## How to run in Google Colab
```
import os
from google.colab import drive
drive.mount('/content/gdrive/')

!pip install tensorboardX
os.chdir('/content/gdrive/My Drive/super-resolution')
!git clone https://github.com/sahebi/map-super-resolution
```
### Train
```
os.chdir('/content/gdrive/My Drive/super-resolution/map-super-resolution')
!python train_all.py --logprefix test_BSDS300_2x -uf 2 --dataset BSDS300 --batchSize 128 --testBatchSize 128 --nEpochs 1500 --iter 3
!python train.py 
        -l  {log_prefix_name} 
        -uf {upscale_factor}
        -ds {dataset_name} 
        -b  {batch_size} 
        -bt {test_batch_size} 
        -n  {epochs}
        -m  {methods}
        -o  {optimizers}

sample:
python train.py -l xxx -uf 1 -ds zurikh -b 4  -bt 8 -n 100 -m vdsr,sub,mem,fsrcnn,srcnn -o adam,sgd,lamb,adadelta,adagrad,rmsprop,rprop,adamax


```

### Test
```
python super_resolve.py --input result/BSD300_3096.jpg/4x/lr.jpg --model model/carn_488.pth --output result/BSD300_3096.jpg/4x/carn.jpg`
```