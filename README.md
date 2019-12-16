# AAGAN
Unofficial PyTorch implementation of Enhanced Single Image Dehazing with Attention-to-Attention Generative Adversarial Network 

This is an unofficial umplementation of the https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8918328

## My changes:

1) In the paper the residual layers have 512 in_channels and out_channels (according to the picture from paper). However, the last enBlock returns 1024 channels. I make residual blocks with 1024 channels.

2) In my opinion, the deBlock on the picture is not the same as the text description from the paper. On picture the first part takes c64, outputs i.e. c64 and the second one outputs c32. But in text it is written that the first one decreases number of channels (i.e. c64 > c32) and the second one leaves number of channles same. The approach from text is correct, otherwise it causes mismatched shapes in DCAM + MCAM interpolation (c32 and c64).

3) Optionally, I have added support of log_softmax instead of softmax, beause it might be more numerically stable. By default it is on. If you think it is incorrect, please contact me.

4) The lr for both generator and discriminator is decreased on every 100k step.

5) I did not perform denosing mentioned in the paper, because I did not find any easy to use implementation of gradient guided image filter. I have used raw dataset with random crop and random horizontal flip.

## Dataset
The project by default uses NYUv2 labeled dataset. Just download dataset .mat file and splits.mat for train/test split and specify the patthes to them when runnig script.

## Requirements
PyTorch 1.3.1
Torchvision
tqdm
scipy
h5py
Tensorboard for visualization
